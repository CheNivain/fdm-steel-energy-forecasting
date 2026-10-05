"""Load either trained model, predict again, and verify the recorded test results."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression_common import load_data, prediction_table
from viva.team_results import print_results, read_predictions

FOLDERS = {"linear": "linear regression", "forest": "random forest"}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_inputs(kind, artifacts=None):
    output_dir = ROOT / FOLDERS[kind] / "outputs_advanced"
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    artifact_dir = Path(artifacts or metadata["input_directory"])
    for filename, expected in metadata["input_sha256"].items():
        path = artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Use --artifacts with the original prepared-data folder.")
        if file_hash(path) != expected:
            raise ValueError(f"{path} differs from the data used for the recorded run.")
    X_train, X_test, y_train, y_test = load_data(artifact_dir)
    return output_dir, metadata, X_train, X_test, y_train, y_test


def verify_predictions(output_dir, frame):
    saved = read_predictions(output_dir / "test_predictions.csv")
    pd.testing.assert_index_equal(frame.index, saved.index)
    np.testing.assert_allclose(frame.actual_kWh, saved.actual_kWh, rtol=0, atol=1e-9)
    np.testing.assert_allclose(frame.persistence_kWh, saved.persistence_kWh, rtol=0, atol=1e-9)
    np.testing.assert_allclose(frame.predicted_kWh, saved.predicted_kWh, rtol=1e-10, atol=1e-8)


def demonstrate(kind, artifacts=None):
    output_dir, metadata, X_train, X_test, y_train, y_test = load_inputs(kind, artifacts)
    print(f"\nLoading trained {FOLDERS[kind]}...", flush=True)
    expected = json.loads((output_dir / "verification.json").read_text())["model_sha256"]
    if file_hash(output_dir / "model.joblib") != expected:
        raise ValueError("The saved model changed since verification.")
    # Only load the trusted model produced by this project, not an unknown pickle.
    model = joblib.load(output_dir / "model.joblib")
    start = time.perf_counter()
    predictions = model.predict(X_test)
    elapsed = time.perf_counter() - start
    frame = prediction_table(y_test, predictions, X_test["lag_1"])
    verify_predictions(output_dir, frame)
    print(f"Training rows: {len(X_train):,}; inputs: {X_test.shape[1]}")
    print("\nFirst five NEW predictions from the loaded model:")
    print(frame.head().to_string(float_format=lambda n: f"{n:.3f}"))
    print_results(frame, metadata["settings"]["algorithm"], threshold=50)
    print(f"Prediction time (not fit time; excludes loading): {elapsed:.2f} s")
    print("New predictions match the recorded results. No .fit() or tuning was run.")
    print("Scope: one-step 15-minute forecasts using previous actual readings.")
    print("Existing preprocessing/interpolation limits and prior test inspection still apply.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["linear", "forest", "both"], default="both")
    parser.add_argument("--artifacts", type=Path, help="Original X_train/y_train/X_test/y_test directory.")
    args = parser.parse_args()
    try:
        for kind in (["linear", "forest"] if args.model == "both" else [args.model]):
            demonstrate(kind, args.artifacts)
    except (ValueError, OSError, AssertionError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
