"""Viva reporting must not silently mix targets, timestamps or high-load rules."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from viva.team_results import (
    assert_same_evaluation, metric_values, read_predictions,
    save_predictions, validate_predictions,
)


class VivaReportingChecks(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "actual_kWh": [10., 50., 80.],
            "predicted_kWh": [12., 40., 75.],
            "persistence_kWh": [8., 10., 50.],
        }, index=pd.date_range("2018-12-02", periods=3, freq="15min", name="timestamp"))

    def test_threshold_uses_actual_and_includes_boundary(self):
        values = metric_values(self.frame.actual_kWh, self.frame.predicted_kWh, 50)
        self.assertEqual(values["HighLoad_MAE"], 7.5)
        self.assertAlmostEqual(values["MAE"], 17 / 3)
        self.assertAlmostEqual(values["RMSE"], np.sqrt(129 / 3))

    def test_no_high_load_is_unavailable_not_zero(self):
        values = metric_values(self.frame.actual_kWh, self.frame.predicted_kWh, 100)
        self.assertTrue(np.isnan(values["HighLoad_MAE"]))

    def test_bad_threshold_rejected(self):
        for threshold in [0, -1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                metric_values(self.frame.actual_kWh, self.frame.predicted_kWh, threshold)

    def test_mismatched_rows_targets_and_baselines_rejected(self):
        assert_same_evaluation([self.frame, self.frame.copy()])
        with self.assertRaisesRegex(ValueError, "timestamps differ"):
            assert_same_evaluation([self.frame, self.frame.iloc[1:]])
        for column in ["actual_kWh", "persistence_kWh"]:
            changed = self.frame.copy()
            changed.iloc[0, changed.columns.get_loc(column)] += 1
            with self.assertRaisesRegex(ValueError, column):
                assert_same_evaluation([self.frame, changed])

    def test_invalid_prediction_files_rejected(self):
        invalids = [self.frame.iloc[[0, 0]], self.frame.iloc[::-1], self.frame.iloc[:0]]
        nonfinite = self.frame.copy()
        nonfinite.iloc[0, 1] = np.nan
        invalids.append(nonfinite)
        for frame in invalids:
            with self.assertRaises(ValueError):
                validate_predictions(frame)

    def test_export_round_trip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            args = [path, self.frame.index, self.frame.actual_kWh,
                    self.frame.predicted_kWh, self.frame.persistence_kWh]
            save_predictions(*args)
            pd.testing.assert_frame_equal(read_predictions(path), self.frame, check_freq=False)
            with self.assertRaises(FileExistsError):
                save_predictions(*args)


if __name__ == "__main__":
    unittest.main()
