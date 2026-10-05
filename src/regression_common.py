"""Shared data checks, metrics and exports for the two regression notebooks."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


VALIDATION_START = pd.Timestamp("2018-09-01")
TEST_START = pd.Timestamp("2018-11-01")
FEATURES = [
    "tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "load_level",
    "lag_1", "lag_4", "lag_96", "lag_672", "roll_mean_4", "roll_std_4",
    "roll_mean_96", "roll_max_96", "diff_1",
]
LIMITATIONS = [
    "Forecasts are one interval ahead (15 minutes), with previous actual "
    "readings available before each prediction. This is not a day-ahead test.",
    "load_level assumes the operating load category is known before the "
    "forecast interval; this must be confirmed with the team.",
    "The supplied preprocessing shifts midnight rows to the next day; "
    "the original interval timestamp convention remains an assumption.",
    "The supplied preprocessing interpolates the target at 2018-11-08 00:00 "
    "using a future reading. This affects that test target and later lag/rolling "
    "features. Test results describe those supplied artifacts, not a fully "
    "causal evaluation of raw measurements.",
    "Upstream feature selection used the entire supplied training period, "
    "including September/October. Validation is provisional conditional on "
    "that fixed feature set; strict validation needs feature selection refitted "
    "on the earlier fitting period.",
    "The preprocessing notebook already inspected the test persistence score "
    "and full-dataset EDA. Do not describe the test set as completely untouched.",
]


def command_line_settings(default_artifacts):
    parser = argparse.ArgumentParser(description="Check inputs or train a regression model.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="Fit, evaluate and save the model.")
    mode.add_argument("--check", action="store_true", help="Check inputs only (the default).")
    parser.add_argument("--artifacts", type=Path, default=default_artifacts)
    return parser.parse_args()


def load_data(artifacts_dir):
    """Load the team's unscaled features and fail on inconsistent inputs."""
    artifacts_dir = Path(artifacts_dir).expanduser().resolve()
    tables = {}
    for name in ["X_train", "y_train", "X_test", "y_test"]:
        path = artifacts_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Set ARTIFACTS_DIR or use --artifacts.")
        table = pd.read_csv(path, index_col=0)
        table.index = pd.to_datetime(table.index, errors="raise")
        table.index.name = "timestamp"
        if table.empty or table.index.hasnans or not table.index.is_unique:
            raise ValueError(f"{name}: empty data, missing timestamps or duplicate timestamps.")
        if not table.index.is_monotonic_increasing:
            raise ValueError(f"{name}: timestamps must already be in chronological order.")
        if not np.isfinite(table.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: missing or infinite values found.")
        tables[name] = table

    X_train, X_test = tables["X_train"], tables["X_test"]
    for name, X in [("train", X_train), ("test", X_test)]:
        target = tables[f"y_{name}"]
        if list(X.columns) != FEATURES:
            raise ValueError(f"X_{name}: expected the team's 15 features in their saved order.")
        if list(target.columns) != ["Usage_kWh"]:
            raise ValueError(f"y_{name}: expected a single Usage_kWh target column.")
        if not X.index.equals(target.index):
            raise ValueError(f"{name}: feature timestamps do not match target timestamps.")
        if (target["Usage_kWh"] < 0).any():
            raise ValueError(f"y_{name}: negative energy readings found.")
    if not (X_train.index.max() < TEST_START <= X_test.index.min()):
        raise ValueError("The team split must place training before 2018-11-01 and test after it.")
    all_index = X_train.index.append(X_test.index)
    if not (all_index.to_series().diff().iloc[1:] == pd.Timedelta(minutes=15)).all():
        raise ValueError("Expected consecutive 15-minute intervals across train and test.")

    y_train = tables["y_train"]["Usage_kWh"]
    y_test = tables["y_test"]["Usage_kWh"]
    combined_y = pd.concat([y_train, y_test])
    combined_X = pd.concat([X_train, X_test])
    if not np.allclose(combined_X["lag_1"].iloc[1:], combined_y.shift(1).iloc[1:]):
        raise ValueError("lag_1 does not match previous actual usage. Supply unscaled artifacts.")
    return X_train, X_test, y_train, y_test


def split_for_validation(X_train, y_train):
    earlier = X_train.index < VALIDATION_START
    if earlier.all() or not earlier.any():
        raise ValueError("Need training records on both sides of 2018-09-01.")
    return X_train.loc[earlier], X_train.loc[~earlier], y_train.loc[earlier], y_train.loc[~earlier]


def describe_data(X_train, X_test, X_fit, X_val):
    for name, X in [("Fit", X_fit), ("Validation", X_val), ("Full training", X_train), ("Test", X_test)]:
        print(f"{name:14} {len(X):6,} rows: {X.index.min()} to {X.index.max()}")
    print(f"Input features: {X_train.shape[1]}; target: Usage_kWh")


def score_predictions(model_name, split, actual, predicted):
    """MAE/RMSE are in kWh; R2 is a score, not percentage accuracy."""
    return {
        "model": model_name,
        "split": split,
        "n_rows": len(actual),
        "MAE_kWh": float(mean_absolute_error(actual, predicted)),
        "RMSE_kWh": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) >= 2 else np.nan,
    }


def prediction_table(actual, predicted, persistence):
    return pd.DataFrame({
        "actual_kWh": actual,
        "predicted_kWh": predicted,
        "persistence_kWh": persistence,
        "residual_kWh": actual - predicted,
    }, index=actual.index)


def save_results(output_dir, model, metrics, validation_trials, validation_predictions,
                 test_predictions, feature_effects, artifacts_dir, settings):
    """Called only after training; no output files are made in check mode."""
    cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib", compress=settings.get("model_compression", 0))
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    validation_trials.to_csv(output_dir / "validation_trials.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv")
    test_predictions.to_csv(output_dir / "test_predictions.csv")
    feature_effects.to_csv(output_dir / "feature_effects.csv", index=False)

    files = [Path(artifacts_dir) / f"{name}.csv" for name in ["X_train", "y_train", "X_test", "y_test"]]
    metadata = {
        "status": "trained",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": "Usage_kWh",
        "horizon_minutes": 15,
        "features": FEATURES,
        "validation_start": settings.get("validation_start", str(VALIDATION_START)),
        "test_start": str(TEST_START),
        "selection_metric": settings.get("selection_metric", "validation RMSE_kWh (lower is better)"),
        "settings": settings,
        "model_parameters": model.get_params(),
        "input_directory": str(Path(artifacts_dir).resolve()),
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "joblib"]},
        "limitations": settings.get("limitations", LIMITATIONS),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    sample = test_predictions.iloc[:96 * 7]
    axes[0].plot(sample.index, sample["actual_kWh"], label="Actual", color="#24634a", linewidth=1)
    axes[0].plot(sample.index, sample["predicted_kWh"], label="Predicted", color="#bc3f55", linewidth=1, alpha=0.8)
    axes[0].set(title="First seven test days: 15-minute forecasts", ylabel="Energy (kWh)")
    axes[0].legend()
    axes[1].scatter(test_predictions["actual_kWh"], test_predictions["predicted_kWh"], s=5, alpha=0.25, color="#24634a")
    low = test_predictions[["actual_kWh", "predicted_kWh"]].min().min()
    high = test_predictions[["actual_kWh", "predicted_kWh"]].max().max()
    axes[1].plot([low, high], [low, high], color="#bc3f55", linestyle="--")
    axes[1].set(title="All test intervals", xlabel="Actual energy (kWh)", ylabel="Predicted energy (kWh)")
    axes[2].hist(test_predictions["residual_kWh"], bins=60, color="#24634a")
    axes[2].axvline(0, color="#bc3f55", linestyle="--")
    axes[2].set(title="Test errors", xlabel="Actual minus predicted (kWh)", ylabel="Intervals")
    fig.savefig(output_dir / "evaluation.png", dpi=150)
    plt.close(fig)
    print(f"Saved trained model, scores, predictions and chart to {output_dir}")
