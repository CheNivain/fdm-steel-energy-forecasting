# SVR Reference: High-Load Metric

Source: `/Users/chelakawijesekara/Downloads/svr_model_train.ipynb`, third code
cell (zero-based cell index 2). The notebook was read, not executed or retrained.
Configuration pickle opcodes were inspected without loading the fitted models.

## Exact definition

```python
HIGH_LOAD_KWH = 50
mask = y_true >= HIGH_LOAD_KWH
mean_absolute_error(y_true[mask], y_pred[mask])
```

HighLoad_MAE measures the average absolute prediction error, in kWh, only for
15-minute intervals whose actual Usage_kWh is at least 50. The selection uses
actual usage, not predicted usage. Exactly 50 is included. It is an evaluation
metric for the regression model, not a separate high/low classification model.

It is not the `Load_Type` / `load_level` category. It also does not measure
electrical current in amperes. kWh measures energy consumed during the interval.

The preceding notebook line prints the training target's 75th percentile
(saved output: 51.41 kWh), but the implemented cutoff is a fixed 50 kWh, not an
automatically calculated percentile. The code does not establish a physical
equipment overload limit.

## Applied to these two models

`src/show_model_results.py` now applies the exact same rule to the original
and optimized Linear Regression and Random Forest saved test predictions and
to the last-interval baseline. It does not retrain or change any model.

`reports/terminal_model_results.txt` contains the shareable results.

## Comparison scope

The SVR notebook reports 28,511 training rows, 5,857 test rows and the same 15
base features. The preprocessing config specifies a 2018-11-01 test boundary.
Its saved last-interval scores match the corresponding local scores to the
displayed precision. These observations support consistency but do not verify
that the friend's data files are byte-for-byte identical; those CSVs were not
part of the supplied SVR files.

Validation strategies differ: the SVR notebook reports five-fold expanding
time-series validation, with four-fold validation on every second training row
during its parameter search. Our optimization selects using September-October
validation. Do not compare those CV and validation scores as the same experiment.
Agree on a common validation protocol for final group model selection.

The SVR notebook fits a scaler within its prediction pipeline and scales the
target via TransformedTargetRegressor. Its final test predictions are clipped at
zero outside the saved model; `svr_final_config.pkl` records `clip_min=0`. A later
integration must preserve that clipping and must not apply the separate
`feature_scaler.pkl` again to a pipeline that already scales its raw inputs.
This observation is from the notebook's code, not a new execution of the SVR.
