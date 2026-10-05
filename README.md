# Steel Industry Energy Consumption: EDA

## Final application: Steel Energy

The selected Random Forest is integrated into a local frontend and Python API.
From this project folder, start it with:

```bash
.venv/bin/python run_app.py --port 8002
```

Open **http://127.0.0.1:8002**. The default Forecast page starts beyond the stored
2018 measurements: **1 January 2019, 00:00-00:15**, under the documented interval-end
timestamp assumption. Click **Generate forecast** to advance. Choose any day and
15 or 30 minutes; skipped intervals are estimated explicitly. Opening the app
loads the saved models and does not retrain them.

The 1,000-tree deployment model was refitted on 33,695 usable full-year examples.
Meter data accepts new actual readings, compares them with immutable predictions,
and offers a separate measured-data-only refit. Development test metrics remain
attached to their original model. Recursive forecasts have no claimed accuracy
guarantee. All seven days are available in Historical replay.

Read `docs/SYSTEM_GUIDE.md` for the beginner demonstration and setup guide, and
`docs/ARCHITECTURE_AND_TESTS.md` for the architecture, API and test instructions.
For a new environment, install `requirements-app.txt`. Browser dependencies are
vendored locally. The original EDA and model-development instructions follow.

Data Miners, IT3051 Fundamentals of Data Mining.

Start with `reports/EDA_REPORT.html` for the charts and actual findings.
`reports/VIVA_NOTES.md` is the short explanation to study.
`notebooks/Steel_Industry_EDA.ipynb` contains the Python analysis and saved outputs.

## Reproduce

The original supplied CSV is copied unchanged to `data/Steel_industry_data.csv`.
The analysis uses pandas, NumPy and Matplotlib. It does not train models.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/eda.py
```

To regenerate and execute the annotated notebook:

```bash
.venv/bin/python src/build_notebook.py
```

In Google Colab, upload the notebook and CSV, then set `STEEL_DATA_CSV` to the
uploaded CSV path using `os.environ` before running the first analysis cell.
Colab normally provides the analysis libraries. If a dependency is missing,
install it there first. Local results were verified using the environment
versions recorded in `reports/eda_findings.json`.

## Outputs

- `reports/EDA_REPORT.html`: standalone report with seven embedded charts.
- `reports/VIVA_NOTES.md`: actual findings and concise explanations.
- `reports/eda_findings.json`: numerical findings, dependency versions and source hash.
- `outputs/figures/`: seven PNG charts.
- `outputs/tables/`: data quality, summary statistics and grouped CSV tables.

## Scope and interpretation

The supplied CSV has midnight rows after 23:45 within each labelled date.
EDA sorts literal timestamps and does not shift dates. Confirm the original
timestamp convention before constructing forecasting lags.

This analysis describes the full supplied file. It does not establish an
untouched final test set. Agree chronological train/validation/test periods
before modelling and fit learned preprocessing only on training data.
Same-interval CO2 and electrical measurements can cause forecasting leakage.
Use Load_Type only if it is independently known before the prediction interval.

EDA is completed by these artifacts. Encoding, lag features, dataset splits,
model fitting and a completed preprocessing pipeline are separate work.

Source identified in the approved proposal:
https://doi.org/10.24432/C52G8C

## Model development

**Latest experiment: advanced five-fold training of both models.** Start with
`reports/ADVANCED_TRAINING_REPORT.md` and each model folder's
`advanced_walkthrough.ipynb`. These executed walkthroughs show real results and
reload the saved models without repeating training. Original results remain in
`outputs`; the first optimization is in `outputs_optimized`; the latest experiment
is in `outputs_advanced`. Previous notebooks and reports document the earlier run.

Print both latest test tables, including high-load errors, without retraining:

```bash
.venv/bin/python src/show_model_results.py
```

Reproduce the latest experiment only when needed:

```bash
.venv/bin/python src/advanced_training.py --model linear --train
.venv/bin/python src/advanced_training.py --model forest --train
.venv/bin/python src/build_advanced_report.py
```

The advanced run screens on the last two time-series folds and selects finalists
by the mean RMSE across five expanding folds. See the report for selection scores,
test scores, high-load performance, assumptions and input sensitivity checks.

The `linear regression/` and `random forest/` folders each contain `model.py`,
an explanatory baseline notebook and a short README. Baseline notebooks have
training off by default. Read `MODEL_COMPARISON_PROTOCOL.md` for the common dates,
metrics and limitations inherited from the team's preprocessing.

To reproduce the earlier optimization (already completed):

```bash
.venv/bin/python "linear regression/optimize.py" --train
.venv/bin/python "random forest/optimize.py" --train
.venv/bin/python src/build_optimization_report.py
```

The optimized Linear Regression uses engineered time and spline features with
ordinary least squares. Random Forest uses a broader parameter search. Candidate
selection uses validation RMSE, followed by a full training-period refit. The
team's supplied preprocessing assumptions remain documented in the report.

Check both setups without training:

```bash
.venv/bin/python "linear regression/model.py" --check
.venv/bin/python "random forest/model.py" --check
```

The model dependencies are included in `requirements.txt`. Regenerate the
unexecuted notebooks from their scripts with:

```bash
.venv/bin/python src/build_model_notebooks.py
```
