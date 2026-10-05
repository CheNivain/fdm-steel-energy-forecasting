"""Predict energy usage from the team's 15 prepared, unscaled input features."""

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimized_regression import checked_features


def predict_file(model_path, features_path, output_path):
    model_path, features_path, output_path = map(Path, [model_path, features_path, output_path])
    if output_path.resolve() in {model_path.resolve(), features_path.resolve()}:
        raise ValueError("The prediction output must not replace an input file or the model.")
    X = pd.read_csv(features_path, index_col=0)
    checked_features(X)
    model = joblib.load(model_path)
    predicted = model.predict(X)
    if not np.isfinite(predicted).all():
        raise ValueError("Model produced a nonfinite prediction.")
    result = pd.DataFrame({"predicted_Usage_kWh": predicted}, index=X.index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = predict_file(args.model, args.features, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(result.head().to_string())
    print(f"Saved {len(result):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
