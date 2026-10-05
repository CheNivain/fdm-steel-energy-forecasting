"""Calculate regression scores from real predictions, without training a model."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


COLUMNS = ["actual_kWh", "predicted_kWh", "persistence_kWh"]


def validate_predictions(frame):
    frame = frame.copy()
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors="raise"), name="timestamp")
    if frame.empty or frame.index.hasnans or not frame.index.is_unique:
        raise ValueError("Need nonempty predictions with unique, valid timestamps.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Predictions must already be in timestamp order; do not shuffle them.")
    frame[COLUMNS] = frame[COLUMNS].astype(float)
    values = frame[COLUMNS].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Predictions, actual readings and naive readings must all be finite.")
    if (frame.actual_kWh < 0).any():
        raise ValueError("Actual energy must be nonnegative; resolve invalid targets first.")
    return frame


def read_predictions(path):
    frame = pd.read_csv(path)
    if "timestamp" not in frame:
        raise ValueError(f"{path}: a timestamp column is required.")
    return validate_predictions(frame.set_index("timestamp"))


def metric_values(actual, predicted, threshold):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The high-load threshold must be a positive, finite kWh value.")
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("Actual and predicted values must be nonempty, aligned 1D arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Actual and predicted values must be finite.")
    high = actual >= threshold
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "HighLoad_MAE": float(mean_absolute_error(actual[high], predicted[high])) if high.any() else np.nan,
    }


def print_results(frame, model_name, threshold):
    frame = validate_predictions(frame)
    rows = {
        "Naive (last interval)": metric_values(frame.actual_kWh, frame.persistence_kWh, threshold),
        model_name: metric_values(frame.actual_kWh, frame.predicted_kWh, threshold),
    }
    print(f"\n{model_name}")
    print(f"Test rows: {len(frame):,} | {frame.index.min()} to {frame.index.max()}")
    print(f"High-load rows: {(frame.actual_kWh >= threshold).sum():,} | actual usage >= {threshold:g} kWh")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda n: f"{n:.3f}", na_rep="N/A"))
    baseline_rmse = rows["Naive (last interval)"]["RMSE"]
    if baseline_rmse > 0:
        improvement = 100 * (1 - rows[model_name]["RMSE"] / baseline_rmse)
        print(f"RMSE improvement over naive: {improvement:.1f}%")
    else:
        print("RMSE improvement over naive: N/A (naive RMSE is zero)")
    print("MAE/RMSE/HighLoad_MAE: kWh. R2 is not percentage accuracy.")
    if not (frame.actual_kWh >= threshold).any():
        print("No high-load observations: HighLoad_MAE is unavailable, not zero.")
    return pd.DataFrame(rows).T


def assert_same_evaluation(frames):
    reference = frames[0]
    for frame in frames[1:]:
        if not reference.index.equals(frame.index):
            raise ValueError("NOT COMPARABLE: test timestamps differ. No rows were filtered or joined.")
        for column in ["actual_kWh", "persistence_kWh"]:
            if not np.allclose(reference[column], frame[column], rtol=0, atol=1e-9):
                raise ValueError(f"NOT COMPARABLE: {column} differs on the same timestamps.")


def save_predictions(path, timestamps, actual, predicted, naive):
    """Export aligned arrays in their existing order; never overwrite an existing file."""
    index = pd.DatetimeIndex(pd.to_datetime(timestamps), name="timestamp")
    arrays = [np.asarray(x, dtype=float) for x in [actual, predicted, naive]]
    if any(x.ndim != 1 or len(x) != len(index) for x in arrays):
        raise ValueError("Every prediction array must match the timestamp count.")
    frame = validate_predictions(pd.DataFrame(dict(zip(COLUMNS, arrays)), index=index))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="x")
    print(f"Saved {len(frame):,} rows to {path.resolve()}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+", help="Dated prediction CSVs, not summary metric CSVs.")
    parser.add_argument("--threshold", type=float, required=True, help="Explicit actual-usage cutoff in kWh.")
    parser.add_argument("--compare", action="store_true", help="Reject different evaluation rows/targets/baselines.")
    args = parser.parse_args()
    try:
        if not np.isfinite(args.threshold) or args.threshold <= 0:
            raise ValueError("The high-load threshold must be a positive, finite kWh value.")
        frames = [read_predictions(path) for path in args.files]
        if args.compare:
            assert_same_evaluation(frames)
        for path, frame in zip(args.files, frames):
            print_results(frame, str(path), args.threshold)
        if args.compare:
            print("\nEvaluation rows, actual values and naive predictions match.")
            print("Training periods, preprocessing, forecast horizon and leakage still require a separate check.")
        elif len(frames) > 1:
            print("\nSeparate reports only: these files have NOT been checked for comparability.")
        print("Scores recalculated from predictions. No model trained. No winner selected.")
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
