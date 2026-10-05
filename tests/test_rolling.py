"""Exercise recursive inputs, reconciliation, immutability and leakage prevention."""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from threading import Event, enumerate as threads

import numpy as np
import pandas as pd

from backend.deployment import seed_measurements, training_frame
from backend.features import InputError, build_features, parse_meter_csv
from backend.rolling import RollingWorkspace, timestamp
from backend.storage import Store


class RecordingModel:
    def __init__(self):
        self.inputs = []

    def predict(self, X):
        self.inputs.append(X.copy())
        return X.lag_1.to_numpy() + 1


class RollingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model = RecordingModel()
        service = SimpleNamespace(store=Store(Path(self.temp.name)/"state.sqlite3"), model=self.model)
        self.workspace = RollingWorkspace(service)
        self.workspace.model = self.model
        self.workspace.model_info = dict(sha256="test-only", training_end="2019-01-01T00:00:00", data_sha256="unset")

    def tearDown(self):
        self.temp.cleanup()

    def forecast(self, start=None, intervals=1, request_id=None):
        state = self.workspace.snapshot()
        return self.workspace.forecast(start or state["next_start"], intervals, 50, state["revision"], request_id or str(uuid4()))

    def test_seed_first_interval_and_recursive_inputs(self):
        state = self.workspace.snapshot()
        self.assertEqual(state["next_start"], "2019-01-01T00:00:00")
        first = self.forecast()["requested"][0]
        self.assertEqual(first["forecast_timestamp"], "2019-01-01T00:15:00")
        self.assertEqual(first["estimated_inputs"], 0)
        second = self.forecast()["requested"][0]
        self.assertEqual(second["estimated_inputs"], 1)
        self.assertEqual(second["input_features"]["lag_1"], first["predicted_kwh"])
        self.assertEqual(second["horizon_steps"], 2)
        self.assertEqual(self.workspace.snapshot()["actual_rows"], 35040)

    def test_jump_to_evening_builds_every_required_interval(self):
        result = self.forecast("2019-01-01T18:00:00", 2)
        self.assertEqual(len(result["forecasts"]), 74)
        self.assertEqual(len(result["requested"]), 2)
        self.assertEqual(result["requested"][0]["estimated_inputs"], 72)
        self.assertEqual(self.workspace.snapshot()["next_start"], "2019-01-01T18:30:00")

    def test_actual_reconciliation_keeps_forecast_and_resets_future_inputs(self):
        result = self.forecast(intervals=2)
        original = result["forecasts"][0]
        history = pd.Series([25.0], index=pd.DatetimeIndex(["2019-01-01T00:15:00"]))
        self.workspace.import_actuals(history, "January.csv")
        ledger = self.workspace.ledger()
        first = next(r for r in ledger if r["id"] == original["id"])
        self.assertEqual(first["predicted_kwh"], original["predicted_kwh"])
        self.assertEqual(first["actual_kwh"], 25)
        self.assertEqual(first["absolute_error_kwh"], abs(25-original["predicted_kwh"]))
        self.assertEqual(self.workspace.snapshot()["next_start"], "2019-01-01T00:15:00")
        new = self.forecast()["requested"][0]
        self.assertEqual(new["input_features"]["lag_1"], 25)
        self.assertEqual(new["estimated_inputs"], 0)
        self.assertEqual(self.workspace.comparison()["all"]["rows"], 1)

    def test_request_id_retries_and_stale_browser_rejected(self):
        identifier = str(uuid4())
        first = self.forecast(request_id=identifier)
        retry = self.workspace.forecast("2019-01-01T00:00:00", 1, 50, 0, identifier)
        self.assertTrue(retry["replayed"])
        self.assertEqual(retry["forecasts"], first["forecasts"])
        self.assertEqual(len(self.workspace.ledger()), 1)
        with self.assertRaises(InputError):
            self.workspace.forecast("2019-01-01T00:15:00", 1, 50, 0, str(uuid4()))
        with self.assertRaises(InputError):
            self.workspace.forecast("2019-01-01T00:15:00", 1, 50, 1, identifier)

    def test_gap_corrections_and_invalid_data_are_rejected_atomically(self):
        for date in ["2017-12-31T23:45:00", "2019-01-02T00:15:00"]:
            with self.assertRaises(InputError):
                self.workspace.import_actuals(pd.Series([10.], index=pd.DatetimeIndex([date])), "bad.csv")
        corrected = pd.Series([99.], index=pd.DatetimeIndex(["2019-01-01T00:00:00"]))
        with self.assertRaises(InputError):
            self.workspace.import_actuals(corrected, "correction.csv")
        result = self.workspace.import_actuals(corrected, "correction.csv", replace=True)
        self.assertEqual(result["imported"]["corrected"], 1)
        self.assertEqual(self.workspace.snapshot()["latest_actual_kwh"], 99)
        for value in ["garbage", "2019-01-01T00:04:00", "2019-01-01T00:15:00Z"]:
            with self.assertRaises(InputError): timestamp(value)
        with self.assertRaises(InputError): self.forecast("2020-01-01T00:00:00")

    def test_forecast_failure_does_not_partially_advance(self):
        with patch.object(self.model, "predict", side_effect=[np.array([2.]), np.array([np.nan])]):
            with self.assertRaises(RuntimeError): self.forecast(intervals=2)
        self.assertEqual(len(self.workspace.ledger()), 0)
        self.assertEqual(self.workspace.revision(), 0)

    def test_storage_persists_and_actuals_only_enter_training(self):
        self.forecast(intervals=2)
        again = RollingWorkspace(self.workspace.service)
        self.assertEqual(again.snapshot()["forecast_count"], 2)
        self.assertEqual(len(again.measurements()), 35040)
        self.assertEqual(again.measurements().index.max(), pd.Timestamp("2019-01-01"))
        self.assertEqual(hashlib.sha256(again.measurements().to_csv().encode()).hexdigest(),
                         hashlib.sha256(seed_measurements().to_csv().encode()).hexdigest())

    def test_first_issued_prediction_not_best_revised_prediction_is_scored(self):
        first = self.forecast(intervals=2)["forecasts"]
        self.workspace.import_actuals(pd.Series([20.], index=pd.DatetimeIndex(["2019-01-01T00:15:00"])), "one.csv")
        self.forecast()
        self.workspace.import_actuals(pd.Series([30.], index=pd.DatetimeIndex(["2019-01-01T00:30:00"])), "two.csv")
        score = self.workspace.comparison()
        self.assertEqual(score["all"]["rows"], 2)
        self.assertEqual(score["all"]["mae"], (abs(first[0]["predicted_kwh"]-20)+abs(first[1]["predicted_kwh"]-30))/2)
        self.assertEqual(score["recursive"]["rows"], 1)

    def test_concurrent_clients_cannot_double_advance_the_same_revision(self):
        def attempt(_):
            try:
                self.workspace.forecast("2019-01-01T00:00:00", 1, 50, 0, str(uuid4()))
                return "saved"
            except InputError:
                return "stale"
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sorted(pool.map(attempt, range(2))), ["saved", "stale"])
        self.assertEqual(len(self.workspace.ledger()), 1)

    def test_background_refit_uses_actuals_and_resets_only_active_estimates(self):
        self.forecast(intervals=2)
        started, release = Event(), Event()
        captured = []
        def fit(reference, measured, directory):
            captured.append(measured)
            started.set()
            release.wait(5)
            return self.model, {**self.workspace.model_info, "fit_seconds": 0.1}
        with patch("backend.rolling.fit_deployment", side_effect=fit):
            self.workspace.start_training()
            self.assertTrue(started.wait(5))
            try:
                with self.assertRaises(InputError):
                    self.workspace.import_actuals(pd.Series([4.], index=pd.DatetimeIndex(["2019-01-01T00:15:00"])), "blocked.csv")
                with self.assertRaises(InputError): self.workspace.start_training()
            finally:
                release.set()
                for thread in threads():
                    if thread.name == "forest-refit": thread.join(5)
        self.assertEqual(self.workspace.job["status"], "complete")
        self.assertEqual(len(captured[0]), 35040)
        self.assertEqual(captured[0].index.max(), pd.Timestamp("2019-01-01"))
        self.assertEqual(self.workspace.snapshot()["estimated_inputs"], 0)
        self.assertEqual(len(self.workspace.ledger()), 2)


class CausalTrainingTests(unittest.TestCase):
    def test_vectorized_training_matches_runtime_and_excludes_failure_windows(self):
        measured = seed_measurements()
        X, y = training_frame(measured)
        self.assertEqual(len(X), 33695)
        bad = measured.index[~measured.valid][0]
        self.assertFalse(((X.index >= bad) & (X.index <= bad + pd.Timedelta(days=7))).any())
        for target in [X.index[0], X.index[1000], X.index[-1]]:
            history = measured.usage.loc[measured.index < target].iloc[-672:]
            np.testing.assert_allclose(build_features(history).iloc[0], X.loc[target], atol=1e-7)
            self.assertEqual(y.loc[target], measured.usage.loc[target])
        modified = measured.copy()
        modified.loc[modified.index > X.index[1000], "usage"] = 999
        changed, _ = training_frame(modified)
        pd.testing.assert_frame_equal(changed.loc[:X.index[1000]], X.loc[:X.index[1000]])

    def test_one_row_actual_csv_allowed_but_not_for_new_replay_history(self):
        csv = b"timestamp,Usage_kWh\n2019-01-01 00:15:00,3.45\n"
        self.assertEqual(len(parse_meter_csv(csv, minimum=1)), 1)
        with self.assertRaises(InputError): parse_meter_csv(csv)


if __name__ == "__main__":
    unittest.main()
