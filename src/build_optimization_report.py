"""Verify fitted artifacts and assemble the Stage 6-8 experiment evidence."""

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

from src.regression_common import FEATURES, LIMITATIONS, load_data, score_predictions


def markdown_table(table, decimals=4):
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.{decimals}f}" if np.isfinite(value) else "N/A"
        return str(value)
    rows = ["| " + " | ".join(map(str, table.columns)) + " |", "| " + " | ".join(["---"] * len(table.columns)) + " |"]
    rows.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in table.itertuples(index=False, name=None))
    return "\n".join(rows)


def build_walkthrough(folder, kind):
    title = "Linear Regression" if kind == "linear" else "Random Forest Regression"
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md(f"# {title}: optimized model and evaluation\n\nThis notebook shows the **actual completed run**. Run All loads saved models and verifies their predictions; it does not start a new search unless you set `RUN_TRAINING = True`. Study the numbered explanations in order. The original baseline notebook and outputs remain in this folder."),
        md("## 1. Import the tools\n`pandas` reads tables, `joblib` loads fitted models, and the shared module contains the optimization code. `X` means input features and `y` means actual energy usage."),
        code('from pathlib import Path\nimport sys\nimport json\nimport joblib\nimport pandas as pd\nimport numpy as np\nfrom IPython.display import display, Image\n\nstart = Path.cwd()\nROOT = next(p for p in [start, *start.parents] if (p / "src" / "optimized_regression.py").is_file())\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\nfrom src.optimized_regression import run, make_estimator\nfrom src.regression_common import load_data, split_for_validation, score_predictions'),
        md("## 2. Set paths and the optional training switch\nTraining is already complete. Keeping this switch False lets you inspect the results quickly during the viva. Setting it True repeats the full optimization and replaces only `outputs_optimized`."),
        code(f'KIND = {kind!r}\nOUTPUT_DIR = ROOT / {folder!r} / "outputs_optimized"\nARTIFACTS_DIR = ROOT.parent / "FDM" / "artifacts"\nRUN_TRAINING = False\nif RUN_TRAINING:\n    run(KIND, ARTIFACTS_DIR, OUTPUT_DIR, jobs=4)'),
        md("## 3. Load the common data and check the dates\nJanuary-August fits candidates. September-October chooses settings. The chosen configuration is then fitted again on January-October before evaluating November onward. Dates stay ordered because this is forecasting."),
        code('X_train, X_test, y_train, y_test = load_data(ARTIFACTS_DIR)\nX_fit, X_val, y_fit, y_val = split_for_validation(X_train, y_train)\ndisplay(pd.DataFrame([{"purpose": name, "rows": len(X), "first": X.index.min(), "last": X.index.max()} for name, X in [("Fit", X_fit), ("Validation", X_val), ("Refit", X_train), ("Test", X_test)]]))'),
        md("## 4. Inspect the chosen settings\nThe configuration was fixed using validation RMSE before this run's test score was calculated. Reading old test scores does not make this an entirely untouched test set. The metadata records the exact settings and assumptions."),
        code('metadata = json.loads((OUTPUT_DIR / "run_metadata.json").read_text())\nsettings = metadata["settings"]\nprint(json.dumps(settings["selected_specification"], indent=2))\nprint("Input columns:", settings["raw_input_features"])\nprint("Columns after feature engineering:", settings["transformed_features"])'),
        md("## 5. Build an untrained copy\nThis cell creates the selected architecture without learning. `.fit(X, y)` is what learns from examples. The final layer remains `LinearRegression` for the linear experiment or `RandomForestRegressor` for the forest. Negative energy predictions are floored at zero by the saved wrapper."),
        code('untrained_model = make_estimator(KIND, settings["selected_specification"])\nprint(untrained_model)'),
        md("## 6. Understand the optimization\nFor Linear Regression, we compared raw inputs, interaction/quadratic features, time-of-day effects and spline basis functions. A spline is a set of smooth curve features; ordinary least squares still learns their coefficients. For Random Forest, we used a seeded random search and then tested 400/800 trees for four promising configurations. More trees are only kept when validation supports them. These searches are finite, not proof of a global optimum."),
        code('trials = pd.read_csv(OUTPUT_DIR / "validation_trials.csv")\nprint("Candidates evaluated:", len(trials))\ndisplay(trials[["trial", "stage", "RMSE_kWh", "MAE_kWh", "R2", "specification"]].head(10))'),
        md("## 7. Compare original and optimized results\nMAE is average absolute error; RMSE gives larger errors more weight. Both are measured in kWh and lower is better. R2 is a score, not an accuracy percentage. Use validation to select settings, and test results to report what happened after fitting."),
        code('baseline = pd.read_csv(OUTPUT_DIR.parent / "outputs" / "metrics.csv")\noptimized = pd.read_csv(OUTPUT_DIR / "metrics.csv")\ncomparison = pd.concat([baseline.assign(version="Original"), optimized.assign(version="Optimized")], ignore_index=True)\ndisplay(comparison[["version", "model", "split", "MAE_kWh", "RMSE_kWh", "R2"]])'),
        md("## 8. Load the final trained model and verify it\nThis file includes fitted preprocessing and prediction rules. It receives the same 15 unscaled inputs as the original model. We compare fresh predictions with the saved CSV to check reproducibility."),
        code('model = joblib.load(OUTPUT_DIR / "model.joblib")\npredictions = model.predict(X_test)\nsaved_predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv", index_col=0)\nnp.testing.assert_allclose(predictions, saved_predictions["predicted_kWh"], rtol=1e-10, atol=1e-8)\ndisplay(pd.DataFrame([score_predictions("Reloaded optimized model", "test", y_test, predictions)]))\nprint("Reload verification passed.")'),
        md("## 9. Inspect predictions and errors\nThe first plot shows actual and predicted usage over seven test days. The diagonal in the second plot is perfect prediction. The last plot shows how far predictions miss the actual readings."),
        code('display(Image(filename=str(OUTPUT_DIR / "evaluation.png")))\ndisplay(saved_predictions.head())'),
        md("## 10. Inspect the learned relationships\nFor the forest, impurity importance describes how inputs helped tree splits. For Linear Regression, coefficients describe weights on engineered features. Neither proves cause and effect. Correlated features make individual values harder to interpret."),
        code('effects = pd.read_csv(OUTPUT_DIR / "feature_effects.csv")\nif "coefficient_standardized_basis" in effects:\n    effects = effects.assign(magnitude=effects["coefficient_standardized_basis"].abs()).sort_values("magnitude", ascending=False)\ndisplay(effects.head(15))'),
        md("## 11. Check robustness and the load-category assumption\nWalk-forward diagnostics fit earlier months and predict later months. These are retrospective checks of settings selected using September-October, so they are not an independent cross-validation estimate. The load ablation checks what happens when `load_level` is removed; it does not establish whether the real system knows the load category in advance."),
        code('display(pd.read_csv(OUTPUT_DIR / "walk_forward_diagnostics.csv"))\ndisplay(pd.read_csv(OUTPUT_DIR / "load_level_ablation.csv"))\nfor limitation in metadata["limitations"]:\n    print("-", limitation)'),
        md("## 12. Explain your contribution\nA useful explanation is: 'I developed a baseline, optimized it using chronological validation, refitted the selected configuration on all training rows, compared MAE/RMSE/R2 against the baseline and persistence, and saved a reproducible model.' Add your actual selected settings and scores. Read `../reports/MODEL_OPTIMIZATION_REPORT.md` for the group handover and remaining tasks.\n\nThe full training implementation is in `../src/optimized_regression.py`. To repeat it from the terminal, run the command below; this Markdown cell does not execute it.\n\n```bash\npython \"" + folder + "/optimize.py\" --train\n```"),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    path = ROOT / folder / "optimized_walkthrough.ipynb"
    NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT / folder)}}).execute()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    print(f"Executed and saved {path}", flush=True)


def main():
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    rows, details = [], {}
    for folder, kind in [("linear regression", "linear"), ("random forest", "forest")]:
        optimized_dir = ROOT / folder / "outputs_optimized"
        meta = json.loads((optimized_dir / "run_metadata.json").read_text())
        X_train, X_test, y_train, y_test = load_data(meta["input_directory"])
        model = joblib.load(optimized_dir / "model.joblib")
        predicted = model.predict(X_test)
        stored = pd.read_csv(optimized_dir / "test_predictions.csv", index_col=0)
        np.testing.assert_allclose(predicted, stored["predicted_kWh"], rtol=1e-10, atol=1e-8)
        current_scores = score_predictions(folder, "test", y_test, predicted)
        for version, directory in [("Original", ROOT / folder / "outputs"), ("Optimized", optimized_dir)]:
            table = pd.read_csv(directory / "metrics.csv")
            for row in table.to_dict("records"):
                if row["model"] != "Persistence" and row["split"] in {"validation", "test"}:
                    rows.append({"Model": folder.title(), "Version": version, "Split": row["split"],
                                 "MAE (kWh)": row["MAE_kWh"], "RMSE (kWh)": row["RMSE_kWh"], "R2": row["R2"]})
            if version == "Optimized":
                selected = table[(table["model"] != "Persistence") & (table["split"] == "test")].iloc[0]
                for metric in ["MAE_kWh", "RMSE_kWh", "R2"]:
                    np.testing.assert_allclose(current_scores[metric], selected[metric], rtol=1e-10)
        source_paths = [ROOT / "src" / "optimized_regression.py", ROOT / "src" / "regression_common.py", ROOT / folder / "optimize.py"]
        verification = {
            "model_reload_predictions_match": True, "metrics_recomputed_match": True,
            "test_rows": len(X_test), "input_features": FEATURES,
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
            "model_sha256": hashlib.sha256((optimized_dir / "model.joblib").read_bytes()).hexdigest(),
        }
        (optimized_dir / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        details[folder] = meta
        print(f"Verified {folder}: saved predictions and metrics match.", flush=True)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    for ax, split in zip(axes, ["validation", "test"]):
        table = comparison[comparison["Split"] == split]
        labels = [f"{row['Model']}\n{row['Version']}" for row in table.to_dict("records")]
        bars = ax.bar(labels, table["RMSE (kWh)"], color=["#8796a5", "#20745a", "#8796a5", "#c45165"])
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(title=f"{split.title()} RMSE (lower is better)", ylabel="kWh", ylim=(0, table["RMSE (kWh)"].max() * 1.16))
        ax.tick_params(axis="x", labelsize=8)
    fig.savefig(report_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    linear = details["linear regression"]["settings"]
    forest = details["random forest"]["settings"]
    validation = comparison[(comparison.Version == "Optimized") & (comparison.Split == "validation")]
    recommended = validation.loc[validation["RMSE (kWh)"].idxmin(), "Model"]
    report = [
        "# Model Optimization: Progress Evaluation 2",
        "Both models are trained and verified. This report covers the user's two models, not the group's final five-model selection. Passing the individual viva also requires understanding and explaining the work.",
        "## Results",
        "Linear Regression here means ordinary least squares with the selected engineered features and a nonnegative prediction floor. It is not a claim that plain linear regression on the original 15 columns achieved the optimized scores. Random Forest also uses its selected feature representation.",
        markdown_table(comparison),
        f"Based on the agreed validation RMSE, the preferred model among these two is **{recommended}**. Compare all five models on the same dates, targets and information before selecting the group winner. Test scores must not drive further tuning.",
        "![Original and optimized RMSE](model_comparison.png)",
        "## What Was Optimized",
        f"Linear Regression: {linear['search_candidates']} candidate input representations; selected `{json.dumps(linear['selected_specification'], sort_keys=True)}`. The final estimator is scikit-learn LinearRegression (ordinary least squares), with {linear['transformed_features']} transformed input columns. Time-specific lag interactions let the relationship between recent usage and the prediction depend on the quarter-hour. Spline basis functions allow smooth curved relationships while coefficients are still learned by least squares. Scaling and spline knots are fitted using fitting rows only. Negative predictions are floored at zero because energy consumption is nonnegative. The original unmodified linear baseline remains in `linear regression/outputs`.",
        f"Random Forest: {forest['search_candidates']} candidates. The search includes the original selected configuration, a deterministic randomized search, and 400/800-tree refinements of four promising configurations. Selected `{json.dumps(forest['selected_specification'], sort_keys=True)}`. It remains a RandomForestRegressor. The original trained forest remains in `random forest/outputs`.",
        "Engineered features reuse the original 15 inputs. Reconstructed lag_2 = lag_1 - diff_1; reconstructed lag_3 = 4 * roll_mean_4 - lag_1 - lag_2 - lag_4. These formulas only use previous observations already represented in the input row. No new current-interval electrical measurement or target is added.",
        "These are bounded searches. More training cannot guarantee the best possible model or first place among the team's algorithms. No Ridge, Lasso or other algorithm is silently renamed Linear Regression.",
        "## Validation and Training",
        "1. Fit candidates on 22,655 January-August rows.\n2. Select using RMSE on the same 5,856 September-October rows for both models. The saved selection file is written before final test evaluation.\n3. Run retrospective walk-forward diagnostics on May-June and July-August. These do not select another winner and are not independent estimates because settings were selected using later data.\n4. Refit the selected configuration on all 28,511 January-October training rows.\n5. Evaluate on 5,857 November onward test rows. Earlier actual readings are available for each one-step prediction.\n6. Reload the saved models and independently recompute their predictions and metrics.",
        "Training on the test answers would remove the meaningful separation between fitting and evaluation. These saved evaluation models deliberately use all training rows, not test targets. R2 is not percentage accuracy. RMSE and MAE are in kWh.",
        "## Baselines and Additional Evidence",
    ]
    for folder in details:
        directory = ROOT / folder / "outputs_optimized"
        report.extend([
            f"### {folder.title()}",
            markdown_table(pd.read_csv(directory / "metrics.csv")),
            "Retrospective walk-forward diagnostics:",
            markdown_table(pd.read_csv(directory / "walk_forward_diagnostics.csv")),
            "Removing the assumed-known load category, with the same selected settings:",
            markdown_table(pd.read_csv(directory / "load_level_ablation.csv")),
        ])
    report.extend([
        "## Limitations to Explain",
        "\n".join(f"- {line}" for line in LIMITATIONS),
        "- Both current models' test results were already inspected before this optimization request. This run selects exclusively by validation, but the overall project test set is not completely untouched.\n- The repeated validation search can itself overfit that period; performance on future meter data can differ.\n- The no-load experiment is a sensitivity check. If Load_Type is not known ahead of time, the team must remove it consistently and rerun the shared comparison.\n- Monthly output has a final January period containing only one interval; R2 is undefined there.\n- The inherited preprocessing issues have been documented, not corrected in this experiment, to retain comparability with the supplied team artifacts.",
        "## Files to Show",
        "- `linear regression/optimized_walkthrough.ipynb` and `random forest/optimized_walkthrough.ipynb`: executed, annotated walkthroughs that load the real trained models.\n- Each `outputs_optimized/model.joblib`: fitted preprocessing, learner and prediction floor.\n- `validation_trials.csv`: every search candidate, validation metrics and elapsed fitting time.\n- `selected_configuration.json`: selected settings fixed before this run's test evaluation.\n- `metrics.csv`, `evaluation.png`, prediction CSVs and `feature_effects.csv`: results and interpretation.\n- `walk_forward_diagnostics.csv` and `load_level_ablation.csv`: robustness evidence.\n- `run_metadata.json` and `verification.json`: input hashes, settings, software versions, source hashes and reload checks.\n- `reports/model_comparison.csv`: original-versus-optimized comparison.",
        "## Repeat Training in VS Code",
        'Open the FDM Project folder, select its .venv interpreter, and run:\n\n```bash\nsource .venv/bin/activate\npython "linear regression/optimize.py" --train\npython "random forest/optimize.py" --train\npython src/build_optimization_report.py\n```\n\nThe first two commands replace only the optimized outputs; baseline outputs remain. The third verifies both artifacts and refreshes this report and the executed walkthrough notebooks. Keep the adjacent FDM/artifacts folder, or pass `--artifacts` when training. For notebook/report reproduction, retain the original artifacts folder used for that run.',
        "## Reuse for the Later System",
        'The prediction entry point accepts prepared input CSVs, not raw steel-industry measurements:\n\n```bash\npython src/predict_regression.py --model "random forest/outputs_optimized/model.joblib" --features "../FDM/artifacts/X_test.csv" --output "tmp/forest_predictions.csv"\n```\n\nKeep the `src` folder with the joblib files because the saved models reference the shared Python classes. The model validates feature names, order, numeric values and missing values. The future backend must maintain past meter readings and build the same features; that backend, frontend and full system integration have not been built in this optimization task.',
        "## Before the Next Evaluation",
        "1. Review the selected settings and your actual validation/test scores.\n2. Understand why the two algorithms suit numerical energy prediction, what each main setting changes, and what MAE/RMSE/R2 mean.\n3. Bring both baseline and optimized results, the experiment table and the saved trained models.\n4. Ask teammates to use the shared dates, target, forecast horizon and metrics; add all their results before choosing the group's final model.\n5. Be ready to describe your own contribution and the preprocessing assumptions honestly.",
        "Assignment reference: IT3051_FDM_Mini_Project_Assignment_2026 (1).pdf, page 3, Stages 6-8. Stage 6 requires at least four suitable algorithms across the group; Stage 7 requires optimization and justified model selection; Stage 8 is an individual evaluation. High scores alone do not guarantee a pass. Backend/frontend and final deliverables appear in later stages.",
    ])
    (report_dir / "MODEL_OPTIMIZATION_REPORT.md").write_text("\n\n".join(report) + "\n")
    for folder, kind in [("linear regression", "linear"), ("random forest", "forest")]:
        build_walkthrough(folder, kind)
    print("Report, comparison chart and both executed walkthroughs are ready.", flush=True)


if __name__ == "__main__":
    main()
