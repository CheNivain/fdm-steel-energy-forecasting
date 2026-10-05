# Linear Regression

Start with `advanced_walkthrough.ipynb` for the latest five-fold experiment.
Its saved trained model and evidence are in `outputs_advanced`. Run
`.venv/bin/python src/show_model_results.py --model linear` from the project root
to see the latest test table without training. Read `reports/ADVANCED_TRAINING_REPORT.md`
for the common CV comparison and the exact selected settings.

`optimized_walkthrough.ipynb` documents the earlier optimized experiment.
It contains actual saved results and explains the process in numbered sections.
`linear_regression.ipynb` is the original baseline. Each numbered section has a short explanation.
`model.py` contains the same code and can also be run as a script.

**Current status: baseline and optimized versions are trained.**
The baseline notebook still defaults to `RUN_TRAINING = False` so opening it does
not repeat training. Baseline results are in `outputs`; optimized results are in
`outputs_optimized`. The latest advanced comparison uses `outputs_advanced`.

## Earlier optimized experiment

The final learner remains ordinary least-squares `LinearRegression`. Eighteen
feature setups were compared using September-October validation RMSE. The
selected version uses quarter-hour-specific lag interactions and spline features,
then scaling and LinearRegression. This is Linear Regression with engineered
features, not plain linear regression on the original inputs. Predictions are
floored at zero and that rule is included in the saved model.

```bash
.venv/bin/python "linear regression/optimize.py" --train
```

Training is already complete; this command is only for reproducing it. The full
implementation is in `src/optimized_regression.py`. Read
`reports/MODEL_OPTIMIZATION_REPORT.md` for scores, evidence and assumptions.
The walkthrough loads the model and verifies predictions without retraining.

## Check the setup

From the FDM Project folder:

```bash
.venv/bin/python "linear regression/model.py" --check
```

Use the project's `.venv/bin/python` as the notebook interpreter in VS Code.
The notebook needs the project folder, including `src/regression_common.py`.
Keep the friend's `FDM/artifacts` folder next to `FDM Project`, or set
`ARTIFACTS_DIR` in the notebook / pass `--artifacts /path/to/artifacts` to the script.

## Reproduce the original baseline

Either change `RUN_TRAINING` to `True` in notebook section 2 and restart/run all,
or run this command (these are alternatives; no need to do both):

```bash
.venv/bin/python "linear regression/model.py" --train
```

Training fits a StandardScaler + LinearRegression pipeline on January-August,
scores September-October, then refits on January-October for the final test.
The scaler is always fitted with the model's fitting rows. There is no iterative
epoch loop and no regularization-strength tuning in ordinary Linear Regression.

Training will create `outputs/model.joblib`, `metrics.csv`,
`validation_trials.csv`, `validation_predictions.csv`, `test_predictions.csv`,
`feature_effects.csv`, `run_metadata.json` and `evaluation.png`.
Rerunning baseline training replaces `outputs` only; optimized outputs remain.

See `../MODEL_COMPARISON_PROTOCOL.md` for shared dates, scoring and preprocessing limitations.
