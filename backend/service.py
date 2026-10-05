"""Trusted model loading, historical evidence and single-interval prediction."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn

from backend.features import InputError, build_features, history_records, from_records
from backend.storage import Store

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "random forest" / "outputs_advanced"
SCENARIOS = {
    "demo-production": ("Monday morning", "2018-11-19 10:00:00"),
    "demo-night": ("Tuesday overnight", "2018-11-20 02:00:00"),
    "demo-wednesday": ("Wednesday morning", "2018-11-21 10:00:00"),
    "demo-thursday": ("Thursday afternoon", "2018-11-22 14:00:00"),
    "demo-friday": ("Friday evening", "2018-11-23 18:00:00"),
    "demo-saturday": ("Saturday morning", "2018-11-24 10:00:00"),
    "demo-weekend": ("Sunday morning", "2018-11-18 10:00:00"),
}


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def metrics(actual, predicted):
    y, p = np.asarray(actual), np.asarray(predicted)
    high = y >= 50
    total = np.sum((y - y.mean()) ** 2)
    return dict(mae=float(np.mean(abs(y - p))), rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                r2=float(1 - np.sum((y - p) ** 2) / total) if total else None,
                high_load_mae=float(np.mean(abs(y[high] - p[high]))) if high.any() else None,
                rows=len(y), high_load_rows=int(high.sum()))


class ForecastService:
    def __init__(self, store_path):
        self.store = Store(store_path)
        self.lock = Lock()
        self.metadata = json.loads((MODEL_DIR / "run_metadata.json").read_text())
        verification = json.loads((MODEL_DIR / "verification.json").read_text())
        self.model_hash = digest(MODEL_DIR / "model.joblib")
        if self.model_hash != verification["model_sha256"]:
            raise RuntimeError("Selected model checksum differs from its verified artifact. Restore the trusted model before starting.")
        if sklearn.__version__ != self.metadata["versions"]["scikit-learn"]:
            raise RuntimeError("scikit-learn version differs from the saved model. Install requirements-app.txt.")
        self.model = joblib.load(MODEL_DIR / "model.joblib")
        if self.model.estimator_.named_steps["features"].use_load:
            raise RuntimeError("This application requires the selected model that ignores the future load category.")
        self.model.estimator_.named_steps["regressor"].n_jobs = 2
        self.raw = self._demo_readings()
        self.evidence = pd.read_csv(MODEL_DIR / "test_predictions.csv", index_col="timestamp", parse_dates=True)
        self.info = self._model_info()

    def _demo_readings(self):
        # Preserve the supplied timestamp convention only for labelled historical demos.
        raw = pd.read_csv(ROOT / "data/Steel_industry_data.csv")
        index = pd.DatetimeIndex(pd.to_datetime(raw["date"], format="%d/%m/%Y %H:%M"))
        index += pd.to_timedelta(((index.hour == 0) & (index.minute == 0)).astype(int), unit="D")
        return pd.Series(raw.Usage_kWh.to_numpy(dtype=float), index=index, name="Usage_kWh")

    def dataset(self, identifier):
        if identifier in SCENARIOS:
            label, target = SCENARIOS[identifier]
            target = pd.Timestamp(target)
            history = self.raw.loc[self.raw.index < target].iloc[-672:]
            return dict(id=identifier, name=label, source="historical_demo", convention="uci",
                        readings=history_records(history), created_utc=None)
        found = self.store.dataset(identifier)
        if found is None:
            raise InputError("This dataset is no longer available. Select a sample or import a CSV.")
        return found

    def describe(self, dataset, include_preview=False):
        history = from_records(dataset["readings"])
        result = {k: v for k, v in dataset.items() if k != "readings"}
        result.update(rows=len(history), start=history.index[0].isoformat(), end=history.index[-1].isoformat(),
                      next_timestamp=(history.index[-1] + pd.Timedelta(minutes=15)).isoformat(),
                      last_kwh=float(history.iloc[-1]), max_kwh=float(history.max()),
                      mean_kwh=float(history.mean()))
        if include_preview:
            result["preview"] = history_records(history.iloc[-96:])
        return result

    def datasets(self):
        return [self.describe(self.dataset(key)) for key in SCENARIOS] + [self.describe(d) for d in self.store.datasets()]

    def forecast(self, identifier, threshold):
        with self.lock:
            dataset = self.dataset(identifier)
            history = from_records(dataset["readings"])
            inputs = build_features(history)
            start = perf_counter()
            prediction = float(self.model.predict(inputs)[0])
            elapsed = perf_counter() - start
            if not np.isfinite(prediction) or prediction < 0:
                raise RuntimeError("Model returned an invalid forecast.")
            target = inputs.index[0]
            baseline = float(history.iloc[-1])
            actual = float(self.raw.loc[target]) if identifier in SCENARIOS else None
            warnings = ["One 15-minute interval only; this is not a daily forecast or a guaranteed upper limit."]
            if dataset["source"] == "historical_demo":
                warnings.append("Historical replay using 2018 observations, not a live factory meter.")
            if dataset["convention"] == "uci":
                warnings.append("Original UCI convention selected: midnight rows are shifted to the next date, matching the team's assumption.")
            if history.max() > 157.18:
                warnings.append("At least one reading exceeds the original dataset maximum (157.18 kWh). Reliability outside the training domain is unknown.")
            result = dict(
                id=str(uuid4()), created_utc=now(), model="Random Forest", model_sha256=self.model_hash,
                dataset_id=identifier, dataset_name=dataset["name"], source=dataset["source"],
                forecast_timestamp=target.isoformat(), horizon_minutes=15, predicted_kwh=prediction,
                previous_kwh=baseline, change_kwh=prediction - baseline,
                alert_threshold_kwh=threshold, high_load=prediction >= threshold,
                actual_kwh=actual, absolute_error_kwh=abs(actual - prediction) if actual is not None else None,
                inference_ms=elapsed * 1000, history_rows_used=672,
                history_sha256=hashlib.sha256(history.iloc[-672:].to_csv().encode()).hexdigest(),
                input_features={k: float(v) for k, v in inputs.iloc[0].items()},
                recent_readings=history_records(history.iloc[-96:]), warnings=warnings,
            )
            self.store.save_forecast(result)
            return result

    def _model_info(self):
        actual = self.evidence.actual_kWh
        scores = metrics(actual, self.evidence.predicted_kWh)
        naive = metrics(actual, self.evidence.persistence_kWh)
        features = pd.read_csv(MODEL_DIR / "feature_effects.csv").head(8).to_dict("records")
        return dict(
            name="Random Forest", version="RF advanced / 02 Oct 2026", model_sha256=self.model_hash,
            metrics=scores, naive=naive, rmse_improvement_pct=100 * (1 - scores["rmse"] / naive["rmse"]),
            high_load_improvement_pct=100 * (1 - scores["high_load_mae"] / naive["high_load_mae"]),
            training_rows=28511, training_start="2018-01-08 00:15:00", training_end="2018-10-31 23:45:00",
            test_start=self.evidence.index[0].isoformat(), test_end=self.evidence.index[-1].isoformat(),
            high_load_threshold_kwh=50, features=features, settings=self.metadata["settings"],
            limitations=self.metadata["limitations"],
            comparisons=[
                dict(model="Random Forest", mae=scores["mae"], rmse=scores["rmse"], r2=scores["r2"], high=scores["high_load_mae"], rows=5857, cutoff=50, group="Local verified"),
                dict(model="Linear Regression", mae=4.128, rmse=7.880, r2=.936, high=9.671, rows=5857, cutoff=50, group="Reported: same test period"),
                dict(model="SVR", mae=4.495, rmse=9.099, r2=.915, high=10.901, rows=5857, cutoff=50, group="Reported: same test period"),
                dict(model="KNN", mae=4.317, rmse=9.185, r2=.887, high=26.187, rows=2880, cutoff=80.411, group="Different test / cutoff"),
                dict(model="Gradient Boosting", mae=3.303, rmse=6.872, r2=.937, high=19.127, rows=2880, cutoff=80.14, group="Different test / cutoff"),
            ],
        )

    def evaluation_day(self, date):
        try:
            day = pd.Timestamp(date)
            if day.tz is not None or day != day.normalize():
                raise ValueError()
            frame = self.evidence.loc[(self.evidence.index >= day) & (self.evidence.index < day + pd.Timedelta(days=1))]
        except (ValueError, TypeError) as exc:
            raise InputError("Select a valid test date.") from exc
        if frame.empty:
            raise InputError("No saved evaluation records on this date. Choose 1 Nov to 31 Dec 2018.")
        return dict(date=date, metrics=metrics(frame.actual_kWh, frame.predicted_kWh),
                    points=[dict(timestamp=t.isoformat(), **{k: float(v) for k, v in row.items()}) for t, row in frame.iterrows()])
