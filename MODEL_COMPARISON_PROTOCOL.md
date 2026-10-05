# Shared model comparison protocol

Status: baseline, optimized and advanced experiments are available.
See `reports/ADVANCED_TRAINING_REPORT.md` for the latest five-fold experiment;
`reports/MODEL_OPTIMIZATION_REPORT.md` documents the earlier run. The group's
winner remains undecided until all five models are compared consistently.

## Latest five-fold comparison

The advanced experiment uses TimeSeriesSplit(n_splits=5) over the same 28,511
training rows, matching the five-fold reporting structure in the supplied SVR
notebook. It screens on folds 4-5, then selects finalists by mean fold RMSE over
all five. The previous optimized settings are always included among the finalists.
`outputs_advanced/fold_boundaries.csv` gives the exact windows. HighLoad_MAE is
the average absolute error where actual Usage_kWh >= 50, independent of Load_Type.

Use this common five-fold protocol for comparing selection scores with teammates.
Do not mix these scores with the older September-October validation score below.
The CV scores were used for tuning and are not independent nested-CV estimates.
All versions retain the same final test timestamps; test scores do not select
the advanced settings. The existing test period has been seen during development.

The advanced input contract remains the same 15 columns. Its transformer may
ignore load_level, depending on selected settings. Learned transformations and
optional weighting use only each fitting fold. The learner remains LinearRegression
or RandomForestRegressor. See the selected configuration for any target-change
formulation or weighting; preserve these inside the saved model during inference.

## The prediction

Predict `Usage_kWh`, energy consumed during one 15-minute interval. At the time
of prediction, the previous interval's actual usage is assumed available. Later
test rows therefore legitimately use earlier test-period actual readings in their
lag features. This is a sequence of one-step forecasts, not a forecast of the whole
test period made on 1 November. A day-ahead system needs a different evaluation.

## Earlier experiment: inputs and dates

Source: `../FDM/artifacts/X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`,
created by your friend's `data_preprocessing.ipynb`. These feature CSVs are unscaled.
The code reads them without changing them and does not load the pickle files.

| Use | Period | Rows |
| --- | --- | ---: |
| Fit candidates | 2018-01-08 00:15 through 2018-08-31 23:45 | 22,655 |
| Validate/select settings | 2018-09-01 00:00 through 2018-10-31 23:45 | 5,856 |
| Refit chosen settings | All supplied training rows, January-October | 28,511 |
| Final test | 2018-11-01 00:00 through 2019-01-01 00:00 | 5,857 |

Both models receive the same 15 feature columns: time/day sine and cosine,
weekend flag, encoded load level, four usage lags, four rolling statistics and
the difference between the previous two usage readings. Timestamps identify rows.

Linear Regression fits its own scaler inside its pipeline. Random Forest uses
the unscaled features. Algorithm-appropriate preprocessing is compatible with
a fair comparison as long as the records, target and available information match.

The optimized versions derive additional features from those same 15 inputs.
Linear Regression uses time-specific lag interactions and learned spline basis
functions, so describe it as "Linear Regression with engineered features".
The forest uses derived calendar, recent-history and relative-usage features.
All learned transformations fit only their fitting period. Both optimized
artifacts include a fixed nonnegative prediction floor. Preserve the plain
baselines for comparison; do not relabel their results as optimized results.

## Earlier single-validation comparison protocol

1. Have all five implementations use the same fitting and validation dates above,
   target and inputs. This protocol proposes RMSE as the primary metric; confirm
   it matches the team's agreed metric before comparing results.
2. Choose settings and the preferred algorithm from validation results. Lower
   RMSE is the primary criterion; also report MAE, R2 and the persistence baseline.
   Validation selection has the upstream feature-selection limitation below.
3. Refit each fixed configuration on the full supplied training set for the
   required final comparison. Use the same final test timestamps and metric code.
4. Report test performance without repeatedly adjusting models to improve it.
   If the team chooses the winner using this test set instead, acknowledge that
   it became a selection set and a further holdout is needed for independent
   confirmation. Do not claim that validation performance guarantees future results.

MAE and RMSE use kWh. R2 is a unitless score, not percentage accuracy. Persistence
predicts `lag_1` and checks whether a learned model adds value over a simple rule.
Store the actual metric values; do not fabricate a percentage accuracy.

## Existing preprocessing assumptions that still apply

- `load_level` comes from the interval's `Load_Type`. It is usable for forecasting
  only if that operating category is independently known ahead of the interval.
  If it is inferred from that interval's measured usage, remove it for every model
  and regenerate consistent artifacts. Current implementations retain the team's
  inputs pending that confirmation.
- Midnight timestamps were shifted to the next day by the preprocessing notebook.
  Confirm the source uses this interval-end convention.
- At 2018-11-08 00:00, the target was interpolated using a future reading. This
  changes one test answer and subsequent lag/rolling inputs. Scores on these
  artifacts should be labelled as results on the existing preprocessing, not
  a fully causal raw-meter test. A strict forecast experiment should regenerate
  features using past-only missing-data handling and score only valid observed
  targets, consistently for all five models.
- Feature selection used all January-October rows, before the new validation
  split. Validation is therefore provisional, conditional on this feature set.
  Strict validation should fit selection using only January-August, then apply
  the chosen selection to validation/test, consistently across all five models.
- The preprocessing already displayed full-data EDA and test baseline scores.
  The supplied test period is held out from these model fits, but it has not been
  completely unseen during the project's development.

## Reusing a saved model later

After training, `joblib.load(".../outputs/model.joblib")` loads the fitted pipeline
or forest. Call `.predict(new_features)` with the same named, ordered input
columns. The artifact expects prepared features, not raw CSV rows. The Linear
Regression pipeline scales its inputs internally. No inference service or
day-ahead forecasting workflow is implemented at this stage.

Use `outputs_optimized/model.joblib` for the optimized version. It includes the
feature transformations and prediction floor and still expects the same 15
unscaled columns. Keep the shared `src` package alongside the model so Python can
import its classes. `src/predict_regression.py` is a checked batch-prediction entry
point for future integration; it does not replace raw-data preprocessing.
