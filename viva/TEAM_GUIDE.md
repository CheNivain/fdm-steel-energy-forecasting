# Team Guide: Show Each Model's Real Results

This guide does not train or change your friends' models. It provides code to
score actual predictions from each person's existing fitted estimator.

## 1. Files to share

Send this guide and `team_results.py`. The scorer is portable and does not need
the other student's model file, project folder or custom model classes.

For local Python, install the scoring dependencies once:

```bash
python -m pip install numpy pandas scikit-learn
```

In Colab these libraries are normally available. Upload `team_results.py` to
the session. In a local notebook, place it beside the notebook and make that
folder the notebook's working directory. Then this import must work:

```python
from team_results import save_predictions, print_results
```

If running in this FDM Project folder instead, use:

```python
from viva.team_results import save_predictions, print_results
```

**Run the chosen model's existing fitting/loading cells first.** A fresh
notebook kernel does not remember a fitted model just because old output is
visible. `NameError: final_model` means the required earlier cell has not run.
To avoid fitting again, each owner should load their own trusted saved model
and recreate exactly its prepared test inputs. Do not load untrusted pickle
files. Retain the fitted pipeline's scaling; do not scale a second time.

## 2. KNN owner's new final cell

This uses the variable names in the supplied KNN notebook, after its fitting
cell. Preserve the existing row order. Do not use the source's full raw date
column for only a test subset.

```python
from team_results import save_predictions, print_results

knn_predictions = final_model.predict(X_test)
knn_report = save_predictions(
    "knn_test_predictions.csv",
    dates_test,
    y_test,
    knn_predictions,
    X_test["lag_1"],
)
print_results(knn_report, "KNN (tuned)", threshold=float(high_thr))
```

This preserves the original training-90th-percentile threshold, about 80.411
kWh for the supplied raw dataset. The screenshot's "top 9%" comes from integer
truncation in its print formatting; `HIGH_LOAD_Q = 0.90` means the 90th
percentile, conventionally the top 10%. This is a display issue, not a new
threshold. The current code includes `Load_Type`; its availability before the
forecast interval must be justified or the feature removed in a revised run.

The original screenshot reports MAE 4.317, RMSE 9.185 and HighLoad_MAE 26.187.
Actual generated predictions determine the new numbers; do not hardcode these.

## 3. SVR owner's new final cell

This uses the previously supplied `svr_model_train.ipynb`, after the final fit.

```python
import numpy as np
from team_results import save_predictions, print_results

assert Xte_final.index.equals(y_test.index)
assert Xte_final.index.equals(X_test.index)
svr_predictions = np.clip(final_model.predict(Xte_final), 0, None)
svr_report = save_predictions(
    "svr_test_predictions.csv",
    y_test.index,
    y_test,
    svr_predictions,
    X_test["lag_1"],
)
print_results(svr_report, "SVR RBF (tuned)", threshold=50)
```

Clipping at zero matches the friend's final evaluation code. The saved SVR
pipeline already scales its inputs. Do not apply `feature_scaler.pkl` again.
The screenshot reports MAE 4.495, RMSE 9.099 and HighLoad_MAE 10.901.

## 4. Gradient Boosting owner's new final cell

Important: the supplied notebook does not calculate HighLoad_MAE, while the
terminal screenshot comes from `gradient_boosting_tuning.py` and does.
The owner should provide that exact script and the processed train/test CSVs
to reproduce the screenshot. They have not been supplied here.

The following works with the **supplied notebook's** variable names, after
its tuning cell, provided its prepared test table includes a previous-usage
column named `usage_lag_1` or `lag_1`. It deliberately checks row alignment.
It adds a documented training-90th-percentile metric; it does not assert that
the missing screenshot script used identical preprocessing or estimator defaults.

```python
from pathlib import Path
import pandas as pd
from team_results import save_predictions, print_results

test_rows = pd.read_csv(Path(PROCESSED_DIR) / "test.csv", parse_dates=["date"])
pd.testing.assert_frame_equal(
    test_rows[feature_cols].reset_index(drop=True),
    X_test.reset_index(drop=True),
    check_dtype=False,
)
pd.testing.assert_series_equal(
    test_rows[TARGET_COL].reset_index(drop=True),
    y_test.reset_index(drop=True),
    check_dtype=False,
)
lag_names = [name for name in ["usage_lag_1", "lag_1"] if name in X_test.columns]
assert len(lag_names) == 1, "Confirm exactly which column is previous actual usage."
gb_predictions = tuned_model.predict(X_test)
gb_threshold = float(y_train.quantile(0.90))
gb_report = save_predictions(
    "gradient_boosting_test_predictions.csv",
    test_rows["date"],
    y_test,
    gb_predictions,
    X_test[lag_names[0]],
)
print_results(gb_report, "Gradient Boosting (tuned)", threshold=gb_threshold)
```

The screenshot reports MAE 3.303, RMSE 6.872, HighLoad_MAE 19.127 and threshold
about 80.14 kWh. These are reference values only; they have not been reproduced
from the supplied notebook/data here. No missing results are invented.

## 5. Repeat a report without fitting or predicting again

The exported CSVs contain timestamps, actual values, model predictions and
naive predictions. They are NOT the summary `*_tuning_results.csv` files.

From the directory containing `team_results.py` and the relevant CSV:

```bash
python team_results.py svr_test_predictions.csv --threshold 50
python team_results.py knn_test_predictions.csv --threshold 80.411
python team_results.py gradient_boosting_test_predictions.csv --threshold 80.14
```

For exact reproduction use the full threshold printed/calculated in your
actual run, not a rounded screenshot threshold. A rounded boundary can change
which rows are included. When exporting again, choose a new filename if the
old CSV exists: the helper intentionally refuses to overwrite evidence.

Alternatively, display a previously exported file inside the notebook:

```python
from team_results import read_predictions, print_results
saved = read_predictions("knn_test_predictions.csv")
print_results(saved, "KNN (saved predictions)", threshold=float(high_thr))
```

## 6. Common threshold does NOT fix different test data

For a deliberately labelled comparison at 50 kWh:

```bash
python team_results.py knn_test_predictions.csv --threshold 50
```

That is a NEW high-load definition for KNN, so its high-load score will differ
from the screenshot. It is still not directly comparable to another model
evaluated on different dates or trained through a different cutoff.

After all five owners agree the evaluation and supply predictions on exactly
the same target timestamps, use:

```bash
python team_results.py linear_test_predictions.csv forest_test_predictions.csv svr_test_predictions.csv knn_test_predictions.csv gradient_boosting_test_predictions.csv --threshold 50 --compare
```

The first two files can be copies of each model's existing
`outputs_advanced/test_predictions.csv`, renamed to the names above.
Currently the KNN/Gradient Boosting results will not pass this shared-row check.
The tool refuses mismatches instead of quietly dropping observations. Matching
rows alone also does not prove matched training cutoffs or absence of leakage.

For legitimate final selection, agree and validate:

1. Forecast horizon, timestamp convention and target cleaning.
2. Training cutoff and chronological validation windows.
3. Inputs actually available before each forecast; algorithm-specific transforms
   may differ, but learned preprocessing must use only the fitting rows.
4. One high-load rule and how fold scores are aggregated; if percentile-based,
   use fitting data only to determine the threshold.
5. The selection objective. Existing searches use RMSE, not HighLoad_MAE.
6. Final reporting without repeatedly tuning against the already seen test data.

Do not re-score a trained model on a period it has already trained on and call
that a test. Do not change the threshold simply to make one model win.

## 7. What each teammate should say

> This is my algorithm, this is why it suits regression, and these are its
> inputs. I fitted it on the earlier training data. I compared settings using
> time-ordered validation. Here are predictions and calculated errors, including
> the naive baseline. My high-load threshold and test dates are shown explicitly.
> I can explain what I changed and the remaining limitations.

For the whole team: currently acknowledge the inconsistent comparison. A
provisional Random Forest recommendation is not a verified five-model winner.

## Verification status

The portable scorer is tested with synthetic fixtures and the local Linear
Regression/Random Forest prediction files. The friend's notebook snippets are
matched to inspected source variable names and syntax-checked, but were not
executed against fitted friend models. Their original files were not modified
or trained. The Gradient Boosting processed data and actual screenshot script
are still needed to verify that run. A summary screenshot cannot regenerate
individual model predictions.
