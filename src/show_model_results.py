"""Print verified saved test results for sharing, without retraining models."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression_common import score_predictions

# Same fixed threshold and inclusive boundary as svr_model_train.ipynb, code cell 3.
HIGH_LOAD_KWH = 50.0


def high_load_mae(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual >= HIGH_LOAD_KWH
    return float(np.abs(actual[mask] - predicted[mask]).mean()) if mask.any() else np.nan


def read_results(directory):
    predictions = pd.read_csv(directory / "test_predictions.csv", index_col=0, parse_dates=True)
    if predictions.empty or not predictions.index.is_unique:
        raise ValueError(f"Missing or duplicate test rows in {directory}.")
    required = ["actual_kWh", "predicted_kWh", "persistence_kWh"]
    if not np.isfinite(predictions[required].to_numpy(dtype=float)).all():
        raise ValueError(f"Invalid saved predictions in {directory}.")
    scores = score_predictions("Saved model", "test", predictions.actual_kWh, predictions.predicted_kWh)
    metrics = pd.read_csv(directory / "metrics.csv")
    saved = metrics.loc[(metrics["split"] == "test") & (metrics["model"] != "Persistence")]
    if len(saved) != 1:
        raise ValueError(f"Expected exactly one saved model test score in {directory}.")
    for name in ["MAE_kWh", "RMSE_kWh", "R2"]:
        np.testing.assert_allclose(scores[name], saved.iloc[0][name], rtol=1e-10, atol=1e-10)
    metadata = json.loads((directory / "run_metadata.json").read_text())
    return predictions, scores, metadata


def format_results(kind, version="latest"):
    folder = "linear regression" if kind == "linear" else "random forest"
    original, original_scores, original_meta = read_results(ROOT / folder / "outputs")
    previous, previous_scores, _ = read_results(ROOT / folder / "outputs_optimized")
    advanced_dir = ROOT / folder / "outputs_advanced"
    use_advanced = version == "advanced" or (version == "latest" and (advanced_dir / "verification.json").is_file())
    final, final_scores, final_meta = read_results(advanced_dir if use_advanced else ROOT / folder / "outputs_optimized")
    pd.testing.assert_index_equal(original.index, final.index)
    np.testing.assert_allclose(original.actual_kWh, final.actual_kWh, rtol=0, atol=1e-10)
    np.testing.assert_allclose(original.persistence_kWh, final.persistence_kWh, rtol=0, atol=1e-10)
    if original_meta["input_sha256"] != final_meta["input_sha256"]:
        raise ValueError("Original and optimized runs used different input artifacts.")
    baseline = score_predictions("Naive", "test", final.actual_kWh, final.persistence_kWh)
    if kind == "linear":
        original_label = "Linear Regression (original)"
        final_label = "Linear Regression (advanced, FINAL)" if use_advanced else "Linear Regression (engineered, FINAL)"
        detail = f"Engineered features: {final_meta['settings']['transformed_features']} (from 15 inputs)"
    else:
        original_label = "Random Forest (original, 8-trial)"
        final_label = "Random Forest (advanced, FINAL)" if use_advanced else "Random Forest (optimized, FINAL)"
        params = final_meta["settings"]["selected_specification"]
        params = params.get("previous", params)
        detail = f"Trees: {params['n_estimators']} | Maximum depth: {params['max_depth']}"
    comparisons = [
        ("Naive (last interval)", baseline, final.persistence_kWh),
        (original_label, original_scores, original.predicted_kWh),
    ]
    if use_advanced:
        pd.testing.assert_index_equal(previous.index, final.index)
        np.testing.assert_allclose(previous.actual_kWh, final.actual_kWh, rtol=0, atol=1e-10)
        comparisons.append(("Previous optimized version", previous_scores, previous.predicted_kWh))
    comparisons.append((final_label, final_scores, final.predicted_kWh))
    rows = []
    for label, scores, predicted in comparisons:
        rows.append({"Model": label, "MAE": scores["MAE_kWh"], "RMSE": scores["RMSE_kWh"],
                     "R2": scores["R2"], "HighLoad_MAE": high_load_mae(final.actual_kWh, predicted)})
    table = pd.DataFrame(rows).set_index("Model")
    table.index.name = None
    improvement = 100 * (baseline["RMSE_kWh"] - final_scores["RMSE_kWh"]) / baseline["RMSE_kWh"]
    fit_seconds = final_meta["settings"]["final_fit_seconds"]
    lines = [
        folder.upper(),
        f"Final model fit time (recorded): {fit_seconds:.2f} s",
        detail,
        f"Test: {len(final):,} intervals | {final.index.min()} to {final.index.max()}",
        f"High-load test intervals: {int((final.actual_kWh >= HIGH_LOAD_KWH).sum()):,} (actual usage >= {HIGH_LOAD_KWH:g} kWh)",
        "",
        table.to_string(float_format=lambda value: f"{value:.3f}"),
        "",
        f"RMSE improvement over naive: {improvement:.1f}%",
    ]
    if use_advanced:
        lines.append(f"Selection score: five-fold mean CV RMSE = {final_meta['settings']['cv_RMSE_kWh']:.3f} kWh")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["both", "linear", "forest"], default="both")
    parser.add_argument("--version", choices=["latest", "optimized", "advanced"], default="latest")
    parser.add_argument("--save", type=Path, help="Also write this summary to a text file.")
    args = parser.parse_args()
    kinds = ["linear", "forest"] if args.model == "both" else [args.model]
    result = "\n\n".join(format_results(kind, args.version) for kind in kinds)
    result += (
        "\n\nMAE and RMSE are in kWh; R2 is not percentage accuracy."
        f"\nHighLoad_MAE: mean absolute error where actual Usage_kWh >= {HIGH_LOAD_KWH:g}, matching the SVR notebook."
        "\nThis is an energy-usage threshold, not the Load_Type category or an electrical current measurement."
        "\nResults use the supplied team preprocessing; documented assumptions still apply."
        "\nModels were selected using validation/CV RMSE; the tables above report test results."
        "\nSaved predictions and metrics verified. No models were retrained.\n"
    )
    print(result, end="")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(result)
        print(f"\nSaved shareable summary: {args.save.resolve()}")


if __name__ == "__main__":
    main()
