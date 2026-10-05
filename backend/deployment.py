"""Refit the selected configuration on actual measurements, never predictions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone

from backend.features import InputError
from src.regression_common import FEATURES

ROOT = Path(__file__).resolve().parents[1]


def seed_measurements():
    raw = pd.read_csv(ROOT / "data/Steel_industry_data.csv")
    index = pd.DatetimeIndex(pd.to_datetime(raw.date, format="%d/%m/%Y %H:%M"))
    index += pd.to_timedelta(((index.hour == 0) & (index.minute == 0)).astype(int), unit="D")
    invalid = (raw.Usage_kWh == 0) & (raw.Lagging_Current_Power_Factor == 0) & (raw.Leading_Current_Power_Factor == 0)
    return pd.DataFrame({"usage": raw.Usage_kWh.to_numpy(float), "valid": ~invalid.to_numpy()}, index=index)


def training_frame(measurements):
    """Match inference lags; exclude targets/windows containing unverified readings."""
    frame = measurements.sort_index().asfreq("15min")
    y = frame.usage.where(frame.valid.eq(True))
    index = frame.index
    angle = 2 * np.pi * (index.hour * 60 + index.minute) / 1440
    day = 2 * np.pi * index.dayofweek / 7
    past = y.shift(1)
    X = pd.DataFrame({
        "tod_sin": np.sin(angle), "tod_cos": np.cos(angle),
        "dow_sin": np.sin(day), "dow_cos": np.cos(day),
        "is_weekend": (index.dayofweek >= 5).astype(int), "load_level": 0,
        "lag_1": past, "lag_4": y.shift(4), "lag_96": y.shift(96), "lag_672": y.shift(672),
        "roll_mean_4": past.rolling(4).mean(), "roll_std_4": past.rolling(4).std(ddof=1),
        "roll_mean_96": past.rolling(96).mean(), "roll_max_96": past.rolling(96).max(),
        "diff_1": past - y.shift(2),
    }, index=index)[FEATURES]
    usable = y.notna() & X.notna().all(axis=1) & past.rolling(672).count().eq(672)
    return X.loc[usable], y.loc[usable]


def fit_deployment(reference, measurements, directory):
    X, y = training_frame(measurements)
    if len(X) < 1000:
        raise InputError("At least 1,000 complete training examples are required after the seven-day warm-up.")
    model = clone(reference)
    model.set_params(estimator__regressor__n_jobs=2)
    started = perf_counter()
    model.fit(X, y)
    probe = model.predict(X.iloc[-10:])
    if not np.isfinite(probe).all() or (probe < 0).any():
        raise RuntimeError("Refitted model failed its prediction check.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"forest-{uuid4().hex}.joblib"
    path = directory / name
    joblib.dump(model, path, compress=3)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = dict(
        filename=name, sha256=checksum, sklearn_version=sklearn.__version__,
        training_rows=len(X), measured_rows=len(measurements),
        training_start=X.index[0].isoformat(), training_end=X.index[-1].isoformat(),
        actuals_through=measurements.index.max().isoformat(),
        excluded_rows=len(measurements) - len(X), fit_seconds=perf_counter() - started,
        data_sha256=hashlib.sha256(measurements.to_csv().encode()).hexdigest(),
        evaluated=False,
        note="Deployment refit with fixed selected settings. Legacy test scores do not evaluate this refit or recursive forecasts.",
        timestamp_convention="Interval end; original UCI midnight rows shifted +1 day (team assumption).",
    )
    temporary = directory / f"manifest-{uuid4().hex}.tmp"
    temporary.write_text(json.dumps(metadata, indent=2))
    os.replace(temporary, directory / "manifest.json")
    return model, metadata


def load_deployment(directory):
    directory = Path(directory)
    manifest = directory / "manifest.json"
    if not manifest.exists():
        return None, None
    metadata = json.loads(manifest.read_text())
    path = directory / metadata["filename"]
    if path.parent.resolve() != directory.resolve():
        raise RuntimeError("Invalid deployment model path.")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != metadata["sha256"] or sklearn.__version__ != metadata["sklearn_version"]:
        raise RuntimeError("Deployment model integrity or scikit-learn version check failed.")
    return joblib.load(path), metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "app_data/deployment"))
    args = parser.parse_args()
    from backend.service import ForecastService
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        reference = ForecastService(Path(temp) / "reference.sqlite3")
        print("Refitting 1,000-tree Random Forest on usable 2018 actual readings...", flush=True)
        _, metadata = fit_deployment(reference.model, seed_measurements(), args.output)
        print(json.dumps(metadata, indent=2), flush=True)
