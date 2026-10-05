"""Build the short, inference-only notebook used for viva rehearsal."""

import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [
        md("# My Viva Practice: Predict and Explain\n\nRun the cells in order. Start with Linear Regression, then switch KIND to forest and run all again. These cells use the trained models; they do not repeat fitting or tuning. Keep the existing data/preprocessing limitations in your explanation. Choose the FDM Project .venv Python kernel."),
        md("## 1. Load the tools\n\n**Say:** Pandas handles tables, NumPy handles numerical arrays, joblib loads the trained model, and scikit-learn provides the error calculations. The path code finds our project so its model classes can be imported."),
        code('from pathlib import Path\nimport sys\nimport json\nimport joblib\nimport numpy as np\nimport pandas as pd\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nfrom IPython.display import display\n\nstart = Path.cwd()\nROOT = next(p for p in [start, *start.parents] if (p / "src" / "advanced_training.py").is_file())\nif str(ROOT) not in sys.path:\n    sys.path.insert(0, str(ROOT))\nfrom viva.live_demo import load_inputs, verify_predictions\nfrom src.regression_common import prediction_table\nprint("Project:", ROOT)'),
        md("## 2. Choose the model and load the inputs\n\n**Say:** X contains the input features, and y contains the actual energy readings. The helper checks that the prepared data matches our recorded experiment. We are not fitting a model here. Change KIND to forest to demonstrate Random Forest."),
        code('KIND = "linear"  # Change to "forest" for Random Forest.\nARTIFACTS = None  # Keep None here; if data was moved, set its original prepared-data folder.\noutput_dir, metadata, X_train, X_test, y_train, y_test = load_inputs(KIND, ARTIFACTS)\nprint("Training rows:", len(X_train))\nprint("Test rows:", len(X_test))\nprint("Input features:", X_test.shape[1])\nprint("Test dates:", X_test.index.min(), "to", X_test.index.max())'),
        md("## 3. Look at the inputs and answers\n\n**Say:** Each row is one 15-minute interval. For example, lag_1 is the previous actual usage and lag_96 is the usage one day earlier. head displays five rows; it does not train anything."),
        code('display(X_test.head())\ndisplay(y_test.head())'),
        md("## 4. Load the trained model and predict\n\n**Say:** joblib.load restores the fitted estimator. predict receives X_test, not y_test. Its saved preprocessing is applied internally. This forest automatically adds its predicted change to previous usage; I must not add it twice. Only load our trusted project artifact."),
        code('model = joblib.load(output_dir / "model.joblib")\npredictions = model.predict(X_test)\nprint("Predictions generated:", len(predictions))\nprint("First five predictions:", predictions[:5])'),
        md("## 5. Type the metric calculation\n\n**Say:** MAE averages absolute errors. RMSE gives more weight to large errors. R2 is not percentage accuracy. The mask selects actual usage of at least 50 kWh for the high-load error. The answers are used for evaluation only after prediction."),
        code('HIGH_LOAD_KWH = 50.0\n\ndef calculate_metrics(actual, predicted):\n    actual = np.asarray(actual)\n    predicted = np.asarray(predicted)\n    mask = actual >= HIGH_LOAD_KWH\n    return {\n        "MAE": mean_absolute_error(actual, predicted),\n        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),\n        "R2": r2_score(actual, predicted),\n        "HighLoad_MAE": mean_absolute_error(actual[mask], predicted[mask]) if mask.any() else np.nan,\n    }\n\nprint("High-load test intervals:", int((y_test >= HIGH_LOAD_KWH).sum()))'),
        md("## 6. Compare with the naive prediction\n\n**Say:** The naive rule copies the previous usage. The table scores the naive and learned predictions on exactly the same rows. Lower error is better. This is a test table, not the cross-validation table used to select settings."),
        code('naive_predictions = X_test["lag_1"].to_numpy()\nresults = pd.DataFrame({\n    "Naive (last interval)": calculate_metrics(y_test, naive_predictions),\n    metadata["settings"]["algorithm"]: calculate_metrics(y_test, predictions),\n}).T\ndisplay(results.round(3))\n\nexamples = prediction_table(y_test, predictions, X_test["lag_1"])\ndisplay(examples.head().round(3))'),
        md("## 7. Verify this is the recorded model\n\n**Say:** I am checking that these newly generated predictions match our saved experiment. Passing this check confirms reproducibility on these inputs; it does not prove there is no leakage or that future errors will be identical."),
        code('verify_predictions(output_dir, examples)\nprint("PASS: predictions, target values and timestamps match the saved experiment.")'),
        md("## 8. Explain how settings were selected\n\n**Say:** Settings were selected by the lowest mean RMSE across five time-series validation folds. High-load MAE was also reported, but it was not the selection objective. These cells only display that evidence."),
        code('print(json.dumps(metadata["settings"]["selected_specification"], indent=2))\nprint("Selection:", metadata["selection_metric"])\nprint("Mean CV RMSE:", round(metadata["settings"]["cv_RMSE_kWh"], 3))\ndisplay(pd.read_csv(output_dir / "fold_boundaries.csv"))\nfor note in metadata["limitations"]:\n    print("-", note)'),
        md('## Optional: demonstrate fitting only when asked\n\nThe following is intentionally a text example, so Run All does not retrain. Type it in a NEW code cell only to practise fitting. Start with KIND = "linear". The forest can take several minutes. It refits the selected configuration in memory, does not repeat the search, and does not overwrite your official model.\n\n```python\nfrom src.advanced_training import make_model\npractice_model = make_model(KIND, metadata["settings"]["selected_specification"], jobs=2)\npractice_model.fit(X_train, y_train)\npractice_predictions = practice_model.predict(X_test)\ndisplay(pd.DataFrame([calculate_metrics(y_test, practice_predictions)]).round(3))\n```\n\n**Say:** fit learns from the training inputs and answers. predict then uses test inputs only. Evaluation compares the predictions with the test answers afterwards. Do not tune repeatedly based on this test result.'),
        md("## Final selection answer\n\nRandom Forest is a leading candidate, not a verified winner across all five. KNN and Gradient Boosting used different test dates and high-load thresholds. First align the evaluation, choose the primary metric on validation data, and acknowledge that existing test results have already been seen. Read START_HERE.md and VIVA_GUIDE.md beside this notebook."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    path = ROOT / "viva" / "VIVA_PRACTICE.ipynb"
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    if args.execute:
        from nbclient import NotebookClient
        NotebookClient(notebook, timeout=180, kernel_name="python3",
                       resources={"metadata": {"path": str(ROOT / "viva")}}).execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    print(f"{'Executed' if args.execute else 'Created'} {path}")


if __name__ == "__main__":
    main()
