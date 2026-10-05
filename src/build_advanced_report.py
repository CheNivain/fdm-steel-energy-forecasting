"""Verify and document the completed five-fold model experiments."""

import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_optimization_report import markdown_table
from src.regression_common import LIMITATIONS, load_data
from src.show_model_results import high_load_mae, read_results


def build_notebook(folder, kind):
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md(f"# {folder.title()}: advanced training\n\nThis is the latest experiment. The earlier baseline and optimized notebooks remain as historical evidence. These cells load and verify the completed run. They do not repeat training."),
        md("## 1. Load Python tools and find the project\nX is the table of inputs, y is the actual energy usage. The saved model contains its own preprocessing."),
        code('from pathlib import Path\nimport sys, json\nimport joblib\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\nstart = Path.cwd()\nROOT = next(p for p in [start, *start.parents] if (p / "src" / "advanced_training.py").is_file())\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\nfrom src.advanced_training import make_model, extended_scores\nfrom src.regression_common import load_data'),
        md("## 2. Load the saved run and its original data\nThere are 15 named inputs at the model boundary. Feature engineering happens inside the model. Its metadata records the exact data directory and settings."),
        code(f'KIND = {kind!r}\nOUTPUT_DIR = ROOT / {folder!r} / "outputs_advanced"\nmetadata = json.loads((OUTPUT_DIR / "run_metadata.json").read_text())\nsettings = metadata["settings"]\nX_train, X_test, y_train, y_test = load_data(metadata["input_directory"])\nprint("Training rows:", len(X_train), "Test rows:", len(X_test))'),
        md("## 3. Explain the validation design\nTimeSeriesSplit creates five expanding training windows. Every validation window occurs after its fitting data. We screen candidates on the last two windows, then compare finalists across all five. The previous optimized configuration is always evaluated as a finalist. These tuning scores are not an independent nested-CV estimate."),
        code('display(pd.read_csv(OUTPUT_DIR / "fold_boundaries.csv"))\ntrials = pd.read_csv(OUTPUT_DIR / "validation_trials.csv")\nprint("Screening trials:", (trials.phase == "screen").sum())\nprint("Full five-fold trials:", (trials.phase == "full_cv").sum())'),
        md("## 4. Understand the candidates\nLinear Regression tests spline shapes, weekend schedules, removing load level and fit-only weighting. It still uses LinearRegression to learn coefficients. Random Forest tests tree settings, calendar features, weighting, and predicting either usage directly or its change since the previous interval. For change prediction, the saved model adds the previous usage back automatically. No SVR is trained."),
        code('display(trials.loc[trials.phase == "full_cv", ["RMSE_kWh", "MAE_kWh", "R2", "HighLoad_MAE", "RMSE_std", "specification"]].sort_values("RMSE_kWh"))'),
        md("## 5. Inspect the chosen configuration\nThe lowest average five-fold RMSE chooses the configuration. MAE and high-load MAE show trade-offs, but the test set does not select the winner. An untrained estimator is only a recipe; .fit(X, y) learns its parameters."),
        code('print(json.dumps(settings["selected_specification"], indent=2))\nprint("Raw inputs:", settings["raw_input_features"], "Engineered inputs:", settings["transformed_features"])\nuntrained = make_model(KIND, settings["selected_specification"])\nprint(untrained)'),
        md("## 6. Inspect the final scores\nThe selected model is refitted on all 28,511 January-October training rows. MAE and RMSE are measured in kWh; lower is better. R2 is not accuracy percent. HighLoad_MAE includes only actual usage >= 50 kWh. cv_mean averages fold metrics, while cv_pooled computes metrics from all out-of-fold predictions together."),
        code('display(pd.read_csv(OUTPUT_DIR / "metrics.csv"))'),
        md("## 7. Verify the exported model\nLoading this file and predicting again must reproduce the saved test results. This checks that preprocessing, target handling and nonnegative clipping were saved correctly."),
        code('model = joblib.load(OUTPUT_DIR / "model.joblib")\npredictions = model.predict(X_test)\nsaved = pd.read_csv(OUTPUT_DIR / "test_predictions.csv", index_col=0)\nnp.testing.assert_allclose(predictions, saved.predicted_kWh, rtol=1e-10, atol=1e-8)\ndisplay(pd.DataFrame([extended_scores("Reloaded model", "test", y_test, predictions)]))\nprint("Saved model verified.")'),
        md("## 8. Read the error plots\nLook at how predictions follow usage changes and where large errors remain. The model is a one-interval-ahead forecast with previous actual meter readings available, not a full day-ahead prediction."),
        code('display(Image(filename=str(OUTPUT_DIR / "evaluation.png")))'),
        md("## 9. Check assumptions and robustness\nThe interpolation sensitivity excludes the known imputed test target and every row whose saved lag/rolling features could depend on it. It does not repair all upstream preprocessing assumptions. The no-load check uses the last CV fold and does not choose a new model."),
        code('display(pd.read_csv(OUTPUT_DIR / "interpolation_sensitivity.csv"))\ndisplay(pd.read_csv(OUTPUT_DIR / "load_level_ablation.csv"))\ndisplay(pd.read_csv(OUTPUT_DIR / "test_monthly_metrics.csv"))\nfor note in metadata["limitations"]:\n    print("-", note)'),
        md("## 10. Reproduce only when needed\nThe experiment is already trained. The command below reruns the search and replaces only this model's outputs_advanced folder.\n\n```bash\npython src/advanced_training.py --model " + kind + " --train\n```\n\nUse `python src/show_model_results.py` for the terminal summary without retraining. Read `../reports/ADVANCED_TRAINING_REPORT.md` for the comparison and remaining limitations."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT / folder)}}).execute()
    nbformat.validate(notebook)
    path = ROOT / folder / "advanced_walkthrough.ipynb"
    nbformat.write(notebook, path)
    print(f"Executed {path}", flush=True)


def main():
    comparison, cv_comparison, details, metadata_by_folder = [], [], [], {}
    for folder in ["linear regression", "random forest"]:
        directory = ROOT / folder / "outputs_advanced"
        frame, scores, metadata = read_results(directory)
        metadata_by_folder[folder] = metadata
        for filename, expected_hash in metadata["input_sha256"].items():
            source = Path(metadata["input_directory"]) / filename
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Training input changed since the saved run: {source}")
        X_train, X_test, y_train, y_test = load_data(metadata["input_directory"])
        model = joblib.load(directory / "model.joblib")
        np.testing.assert_allclose(model.predict(X_test), frame.predicted_kWh, rtol=1e-10, atol=1e-8)
        for version, output in [("Original", "outputs"), ("Previous optimized", "outputs_optimized"), ("Advanced", "outputs_advanced")]:
            p, s, meta = read_results(ROOT / folder / output)
            if meta["input_sha256"] != metadata["input_sha256"]:
                raise ValueError("Different input artifacts cannot share this comparison table.")
            pd.testing.assert_index_equal(p.index, frame.index)
            np.testing.assert_allclose(p.actual_kWh, y_test, atol=1e-10, rtol=0)
            comparison.append({"Model": folder.title(), "Version": version,
                               "MAE": s["MAE_kWh"], "RMSE": s["RMSE_kWh"], "R2": s["R2"],
                               "HighLoad_MAE": high_load_mae(p.actual_kWh, p.predicted_kWh)})
        trials = pd.read_csv(directory / "validation_trials.csv")
        full = trials.loc[trials.phase == "full_cv"].sort_values("RMSE_kWh", kind="stable")
        incumbent = next(row for row in full.to_dict("records") if "previous" in json.loads(row["specification"]))
        best = full.iloc[0]
        cv_tradeoffs = []
        for metric in ["RMSE_kWh", "MAE_kWh", "HighLoad_MAE"]:
            change = float(best[metric] - incumbent[metric])
            cv_tradeoffs.append(f"{metric}: {incumbent[metric]:.4f} to {best[metric]:.4f} ({'lower' if change < 0 else 'higher' if change > 0 else 'unchanged'})")
        for label, row in [("Previous settings", incumbent), ("Selected settings", best)]:
            cv_comparison.append({"Model": folder.title(), "Version": label, "Mean CV RMSE": row["RMSE_kWh"],
                                  "Mean CV MAE": row["MAE_kWh"], "Mean CV HighLoad_MAE": row["HighLoad_MAE"],
                                  "Fold RMSE SD": row["RMSE_std"]})
        settings = metadata["settings"]
        representation_note = (
            "This forest learns Usage_kWh minus lag_1. Its saved prediction method adds lag_1 back, then floors the result at zero. Feature importance describes tree splits for the change target."
            if settings.get("target_mode") == "delta" else
            "This model learns Usage_kWh directly and floors predictions at zero."
        )
        details.extend([
            f"## {folder.title()}",
            f"Screened {settings['screen_candidates']} configurations on the last two CV folds, then evaluated {settings['full_cv_candidates']} finalists on all five folds. Selected `{json.dumps(settings['selected_specification'], sort_keys=True)}`. The final fit uses {settings['final_fit_rows']:,} rows and {settings['transformed_features']} engineered columns.",
            "Cross-validation trade-offs versus the previous settings: " + "; ".join(cv_tradeoffs) + ". RMSE is the selection objective; a worse secondary metric is retained and disclosed.",
            representation_note,
            markdown_table(pd.read_csv(directory / "metrics.csv")),
            "Finalists (same five folds):",
            markdown_table(full[["RMSE_kWh", "MAE_kWh", "HighLoad_MAE", "RMSE_std", "specification"]]),
            "Sensitivity excluding the known interpolated answer and its dependent lag/rolling rows:",
            markdown_table(pd.read_csv(directory / "interpolation_sensitivity.csv")),
            "Last-fold sensitivity when ignoring load_level (identical setting if already ignored):",
            markdown_table(pd.read_csv(directory / "load_level_ablation.csv")),
            "Model export and timing:",
            "```json\n" + (directory / "verification.json").read_text().strip() + "\n```",
        ])
        print(f"Verified advanced {folder} in a separate process.", flush=True)

    test = pd.DataFrame(comparison)
    cv = pd.DataFrame(cv_comparison)
    test.to_csv(ROOT / "reports" / "advanced_model_comparison.csv", index=False)
    cv.to_csv(ROOT / "reports" / "advanced_cv_comparison.csv", index=False)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, metric in zip(axes, ["RMSE", "HighLoad_MAE"]):
        labels = [f"{r.Model}\n{r.Version}" for r in test.itertuples()]
        bars = ax.bar(labels, test[metric], color=["#8796a5", "#7db59d", "#20745a", "#8796a5", "#df8e9b", "#c45165"])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set(title=f"Test {metric} (lower is better)", ylabel="kWh", ylim=(0, test[metric].max() * 1.17))
        ax.tick_params(axis="x", labelsize=7, rotation=25)
    fig.savefig(ROOT / "reports" / "advanced_model_comparison.png", dpi=160)
    plt.close(fig)
    report = [
        "# Advanced Training: Linear Regression and Random Forest",
        "Only these two models were developed and trained. The friend's SVR files were not changed or trained. Earlier outputs and notebooks remain available. These are the best configurations among the evaluated five-fold finalists, not a guarantee of global optimality or best performance on every metric.",
        "## Common Cross-Validation Comparison",
        "TimeSeriesSplit(n_splits=5) uses the same 28,511 chronologically ordered training rows and fold boundaries as the supplied SVR notebook's five-fold reporting stage. Candidates first screen on the last two folds; finalists and the previous optimized configuration are evaluated on all five. The lowest mean fold RMSE selects the configuration. MAE, R2 and high-load MAE are recorded, not silently substituted as the selection objective.",
        markdown_table(cv),
        "The five-fold mean is a tuning score, not an unbiased independent estimate: candidate selection is not nested. Earlier feature choices were informed by the supplied full training period. No model is selected using this run's test scores.",
        "## Final Test Comparison",
        markdown_table(test),
        "![Test errors across versions](advanced_model_comparison.png)",
        "All test comparisons use the same 5,857 target timestamps and input artifacts. HighLoad_MAE uses actual Usage_kWh >= 50, exactly as the friend's notebook. These are 15-minute one-step forecasts with earlier actual readings available. R2 is not percentage accuracy. A configuration that wins cross-validation can still have a worse test metric; report that honestly rather than choosing another configuration by test score.",
        "## What Was Investigated",
        "- LinearRegression: spline knot count/spacing/degree, quarter-hour and weekend interactions, additional change features, removing load_level, higher fitting weights for high-load examples, and recency weighting. The learner remains ordinary or weighted least squares according to the selected configuration.\n- RandomForestRegressor: depth, feature fraction, leaf/split sizes, bootstrap sample fraction, number of trees, schedule features, removing load_level, fit-only weights, and direct versus change prediction. A change model adds lag_1 back before clipping to zero.\n- Random Forest finalists use 400 trees, followed by 1,000-tree checks for the two best new finalists. The previous 800-tree configuration is always a full-CV candidate. Increasing tree count is an experiment, not an automatic improvement.\n- No Ridge, Lasso, boosting, SVR or cross-family ensemble is substituted for the assigned learners.",
        *details,
        "## Data and Interpretation Limits",
        "\n".join(f"- {note}" for note in LIMITATIONS if not note.startswith("load_level")),
        "- Both previous test results were already seen before this request. The project has no fresh untouched future holdout; independent confirmation needs new chronological observations.\n- The known imputed reading and all 98 directly affected answer/feature rows are excluded in a separate sensitivity table. This table does not establish that the full supplied preprocessing is causal.\n- If a selected configuration omits load_level, its 15-column input contract remains unchanged but the transformer ignores that column. A finite placeholder can fill that field when unavailable. Other selected configurations require the load category to be known in advance.\n- Hyperparameter search is finite and can overfit validation. The final fitted artifact is not trained on test targets.\n- These artifacts are models and experiment evidence. They do not implement the complete raw-meter backend, frontend, or guarantee an assessment outcome.",
        "## Files and Commands",
        "Each model's `outputs_advanced` directory contains model.joblib, metrics.csv, validation_trials.csv, cv_fold_trials.csv, fold_boundaries.csv, validation_predictions.csv, test_predictions.csv, evaluation.png, feature_effects.csv, selected_configuration.json, search_protocol.json, interpolation_sensitivity.csv, load_level_ablation.csv and verification.json. The executed `advanced_walkthrough.ipynb` is the current walkthrough; `optimized_walkthrough.ipynb` documents the previous experiment.",
        "Show results without training:\n\n```bash\n.venv/bin/python src/show_model_results.py\n```\n\nRepeat advanced training only when needed:\n\n```bash\n.venv/bin/python src/advanced_training.py --model linear --train\n.venv/bin/python src/advanced_training.py --model forest --train\n.venv/bin/python src/build_advanced_report.py\n```",
        "For prediction, use src/predict_regression.py with the selected outputs_advanced/model.joblib and the same prepared 15-column CSV schema. Retain the src package with the model files. All preprocessing, target handling and nonnegative clipping are included in the saved estimator.",
        "The group's final selection still requires all five algorithms on an agreed validation protocol and documentation of their test results. Do not mix the previous single September-October validation score with the new five-fold mean.",
    ]
    (ROOT / "reports" / "ADVANCED_TRAINING_REPORT.md").write_text("\n\n".join(report) + "\n")
    for folder, kind in [("linear regression", "linear"), ("random forest", "forest")]:
        build_notebook(folder, kind)
    print("Advanced comparison, report and executed notebooks are ready.", flush=True)


if __name__ == "__main__":
    main()
