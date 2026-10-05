"""Run with: .venv/bin/python -m unittest tests/test_application.py -v"""

import io
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from backend.features import InputError, build_features, from_records, parse_meter_csv
from backend.main import create_app
from backend.service import ForecastService, ROOT
from backend.storage import Store


def sample_csv(rows=672):
    index = pd.date_range("2026-09-01", periods=rows, freq="15min")
    return pd.DataFrame({"timestamp": index, "Usage_kWh": np.arange(rows) % 100 + 1.0}).to_csv(index=False).encode()


class MeterFeatureTests(unittest.TestCase):
    def test_exact_lags_and_rolling_sample_std(self):
        history = parse_meter_csv(sample_csv())
        row = build_features(history).iloc[0]
        self.assertEqual(row.lag_1, history.iloc[-1])
        self.assertEqual(row.lag_4, history.iloc[-4])
        self.assertEqual(row.lag_96, history.iloc[-96])
        self.assertEqual(row.lag_672, history.iloc[0])
        self.assertEqual(row.roll_mean_4, history.iloc[-4:].mean())
        self.assertEqual(row.roll_std_4, history.iloc[-4:].std(ddof=1))
        self.assertEqual(row.diff_1, history.iloc[-1] - history.iloc[-2])
        self.assertEqual(row.load_level, 0)
        self.assertEqual(build_features(history).index[0], history.index[-1] + pd.Timedelta(minutes=15))

    def test_old_history_does_not_change_features(self):
        history = parse_meter_csv(sample_csv(1000))
        altered = history.copy()
        altered.iloc[:-672] = 999
        pd.testing.assert_frame_equal(build_features(history), build_features(altered))

    def test_invalid_csv_cases(self):
        frame = pd.read_csv(io.BytesIO(sample_csv()))
        cases = [b"", b"\xff", b"timestamp,Usage_kWh,Usage_kWh\nx,2,3", b"date,energy\nx,1", sample_csv(671)]
        for column, value in [("timestamp", "bad"), ("timestamp", frame.timestamp.iloc[1]), ("timestamp", "2026-09-01 00:03:00"), ("timestamp", "2026-09-01T00:00:00Z"), ("Usage_kWh", -1), ("Usage_kWh", "NaN"), ("Usage_kWh", "inf"), ("Usage_kWh", "")]:
            invalid = frame.copy().astype(object)
            invalid.loc[0, column] = value
            cases.append(invalid.to_csv(index=False).encode())
        cases.append(frame.drop(index=400).to_csv(index=False).encode())
        cases.append(frame.iloc[::-1].to_csv(index=False).encode())
        for content in cases:
            with self.subTest(content=content[:80]), self.assertRaises(InputError):
                parse_meter_csv(content)

    def test_iso_midnight_is_not_shifted(self):
        history = parse_meter_csv(sample_csv())
        self.assertEqual(history.index[0], pd.Timestamp("2026-09-01 00:00"))

    def test_legacy_convention_is_explicit(self):
        frame = pd.read_csv(io.BytesIO(sample_csv()))
        times = pd.to_datetime(frame.timestamp)
        times.loc[times.dt.hour.eq(0) & times.dt.minute.eq(0)] -= pd.Timedelta(days=1)
        frame["date"] = times.dt.strftime("%d/%m/%Y %H:%M")
        history = parse_meter_csv(frame.drop(columns="timestamp").to_csv(index=False).encode(), "uci")
        self.assertEqual(history.index[0], pd.Timestamp("2026-09-01"))
        with self.assertRaises(InputError):
            parse_meter_csv((ROOT / "data/Steel_industry_data.csv").read_bytes(), "uci")

    def test_zero_energy_alone_is_valid(self):
        frame = pd.read_csv(io.BytesIO(sample_csv()))
        frame["Usage_kWh"] = 0
        history = parse_meter_csv(frame.to_csv(index=False).encode())
        self.assertEqual(build_features(history).iloc[0].roll_std_4, 0)


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.service = ForecastService(Path(cls.temp.name) / "test.sqlite3")
        cls.client_context = TestClient(create_app(cls.service))
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        cls.temp.cleanup()

    def upload(self, name="meter.csv"):
        result = self.client.post("/api/datasets", files={"file": (name, sample_csv(), "text/csv")})
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()

    def test_health_static_schema_and_metrics(self):
        self.assertEqual(self.client.get("/api/health").json()["status"], "ready")
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/vendor/chart.js").status_code, 200)
        self.assertEqual(self.client.get("/api/openapi.json").status_code, 200)
        model = self.client.get("/api/model").json()
        self.assertAlmostEqual(model["metrics"]["mae"], 3.64450081144)
        self.assertAlmostEqual(model["metrics"]["high_load_mae"], 9.51498069699)
        self.assertEqual(len(model["comparisons"]), 5)

    def test_demo_predictions_match_saved_evaluation(self):
        for identifier in ["demo-production", "demo-weekend", "demo-night"]:
            response = self.client.post("/api/forecasts", json={"dataset_id": identifier})
            self.assertEqual(response.status_code, 201, response.text)
            result = response.json()
            saved = self.service.evidence.loc[pd.Timestamp(result["forecast_timestamp"])]
            self.assertAlmostEqual(result["predicted_kwh"], saved.predicted_kWh, places=6)
            self.assertEqual(result["actual_kwh"], saved.actual_kWh)
            self.assertEqual(result["source"], "historical_demo")
            self.assertEqual(len(result["input_features"]), 15)
            self.assertEqual(self.client.get(f'/api/forecasts/{result["id"]}').json(), result)

    def test_full_day_raw_history_matches_saved_predictions(self):
        timestamps = pd.date_range("2018-11-19", periods=96, freq="15min")
        inputs = pd.concat([
            build_features(self.service.raw.loc[self.service.raw.index < t].iloc[-672:])
            for t in timestamps
        ])
        predictions = self.service.model.predict(inputs)
        expected = self.service.evidence.loc[timestamps, "predicted_kWh"].to_numpy()
        # Window-local sample std differs slightly from long-series rolling numerics.
        np.testing.assert_allclose(predictions, expected, rtol=0, atol=0.0001)

    def test_upload_append_predict_delete_retains_audit(self):
        dataset = self.upload()
        identifier = dataset["id"]
        response = self.client.post("/api/forecasts", json={"dataset_id": identifier, "alert_threshold_kwh": 1})
        self.assertEqual(response.status_code, 201, response.text)
        forecast = response.json()
        self.assertIsNone(forecast["actual_kwh"])
        self.assertTrue(forecast["high_load"])
        update = self.client.post(f"/api/datasets/{identifier}/readings", json={"timestamp":dataset["next_timestamp"],"usage_kwh":72})
        self.assertEqual(update.status_code, 200, update.text)
        self.assertEqual(update.json()["rows"], 673)
        again = self.client.post(f"/api/datasets/{identifier}/readings", json={"timestamp":dataset["next_timestamp"],"usage_kwh":72})
        self.assertEqual(again.status_code, 422)
        self.assertEqual(self.client.delete(f"/api/datasets/{identifier}").status_code, 204)
        self.assertEqual(self.client.get(f'/api/forecasts/{forecast["id"]}').status_code, 200)
        self.assertEqual(self.client.get(f"/api/datasets/{identifier}").status_code, 422)

    def test_validation_and_origin_guards(self):
        for body in [{}, {"dataset_id":"missing"}, {"dataset_id":"demo-night","alert_threshold_kwh":0}, {"dataset_id":"demo-night","alert_threshold_kwh":True}, {"dataset_id":"demo-night","extra":2}]:
            response = self.client.post("/api/forecasts", json=body)
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.client.post("/api/datasets", files={"file":("bad.csv",b"x,y\n1,2")}).status_code, 422)
        self.assertEqual(self.client.delete("/api/datasets/demo-night").status_code, 422)
        response = self.client.post("/api/forecasts", json={"dataset_id":"demo-night"}, headers={"Origin":"https://example.com"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/api/health", headers={"Host":"evil.example"}).status_code, 400)
        response = self.client.post("/api/datasets", content=b"x",headers={"Content-Length":str(9*1024*1024)})
        self.assertEqual(response.status_code, 413)

    def test_exports_persistence_and_evaluation(self):
        template = self.client.get("/api/template.csv")
        self.assertEqual(len(parse_meter_csv(template.content)), 672)
        self.assertEqual(self.client.get("/api/evaluation?date=2018-11-19").json()["metrics"]["rows"], 96)
        self.assertEqual(self.client.get("/api/evaluation?date=bad").status_code, 422)
        self.assertEqual(self.client.get("/api/evaluation?date=2026-01-01").status_code, 422)
        for path in ["/api/evaluation/export.csv", "/api/metrics/export.csv", "/api/forecasts/export.csv"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn("attachment", response.headers["content-disposition"])
        self.assertEqual(len(Store(self.service.store.path).forecasts()), len(self.service.store.forecasts()))


if __name__ == "__main__":
    unittest.main()
