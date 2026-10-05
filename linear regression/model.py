# %% [markdown]
# # Linear Regression: Steel Industry Energy Consumption
# Predict energy usage (`Usage_kWh`) for one 15-minute interval using calendar
# information and past usage. This notebook starts in **check-only mode**.
# Read the code first. Training will happen later when `RUN_TRAINING` is True.

# %% [markdown]
# ## 1. Import the tools
# pandas handles tables. scikit-learn supplies the model and preprocessing.
# The shared helper file checks the inputs and saves results for both models.

# %%
from pathlib import Path
import os
import sys
from time import perf_counter

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next((p for p in [start, *start.parents] if (p / "src" / "regression_common.py").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Open this notebook from inside the FDM Project folder.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.regression_common import (
    LIMITATIONS, command_line_settings, describe_data, load_data,
    prediction_table, save_results, score_predictions, split_for_validation,
)

# %% [markdown]
# ## 2. Set the paths and training switch
# `False` means check the data and create an untrained model only.
# Later, change it to `True` and run the notebook from the beginning to train.
# When using the script, the equivalent command is `model.py --train`.

# %%
RUN_TRAINING = False
ARTIFACTS_DIR = Path(os.environ.get("FDM_ARTIFACTS", str(ROOT.parent / "FDM" / "artifacts")))
OUTPUT_DIR = ROOT / "linear regression" / "outputs"
MODEL_NAME = "Linear Regression"

if "__file__" in globals() and __name__ == "__main__":
    args = command_line_settings(ARTIFACTS_DIR)
    RUN_TRAINING = args.train
    ARTIFACTS_DIR = args.artifacts

# %% [markdown]
# ## 3. Load X (inputs) and y (answers)
# These are your team's saved, **unscaled** features. The timestamp identifies
# a row but is not itself a model input. The helper checks order, alignment,
# missing values and the previous-interval feature.

# %%
X_train, X_test, y_train, y_test = load_data(ARTIFACTS_DIR)
print(X_train.head(3).to_string())
print(y_train.head(3).to_string())

# %% [markdown]
# ## 4. Set aside validation data
# Fit with January-August records. Use September-October to measure validation
# performance. November onward is the supplied final test period.
# Dates stay in order; there is no random shuffling.

# %%
X_fit, X_val, y_fit, y_val = split_for_validation(X_train, y_train)
describe_data(X_train, X_test, X_fit, X_val)

# %% [markdown]
# ## 5. Build the model (no learning yet)
# Linear Regression learns a weighted sum of the inputs plus an intercept.
# The pipeline learns scaling from the same rows used to fit the model.
# Scaling helps numerical conditioning; ordinary least-squares regression
# does not generally require it. We do not reuse the friend's fitted scaler.
# This model has no tree depth or number-of-trees settings to tune.

# %%
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression()),
])
print(model)
if not RUN_TRAINING:
    print("CHECK COMPLETE. The model is built but untrained. Training cells below are skipped.")

# %% [markdown]
# ## 6. Train on the fitting period and predict validation usage
# `.fit(X_fit, y_fit)` is the instruction that makes the model learn.
# `.predict(X_val)` then estimates usage for the later validation-period rows.
# A persistence baseline simply uses the previous interval's actual usage.

# %%
if RUN_TRAINING:
    started = perf_counter()
    model.fit(X_fit, y_fit)
    validation_prediction = model.predict(X_val)
    validation_seconds = perf_counter() - started
    validation_metrics = pd.DataFrame([
        score_predictions(MODEL_NAME, "validation", y_val, validation_prediction),
        score_predictions("Persistence", "validation", y_val, X_val["lag_1"]),
    ])
    print(validation_metrics.to_string(index=False))

# %% [markdown]
# ## 7. Refit the fixed model using all training rows
# After assessing this fixed Linear Regression configuration, create a fresh
# copy and fit January-October. `clone` copies settings, not learned weights.
# Ordinary Linear Regression uses its standard intercept; we do not invent
# a tuning search or switch to Ridge/Lasso, which are different estimators.

# %%
if RUN_TRAINING:
    final_model = clone(model)
    started = perf_counter()
    final_model.fit(X_train, y_train)
    final_fit_seconds = perf_counter() - started

# %% [markdown]
# ## 8. Evaluate the final model
# MAE is the average absolute error in kWh. RMSE gives larger errors more
# weight. Lower is better for both. Higher R2 is better; R2 is not accuracy
# expressed as a percentage and can be negative. No predictions are clipped.
# Use validation RMSE to compare models/settings. Do not tune using test scores.

# %%
if RUN_TRAINING:
    train_prediction = final_model.predict(X_train)
    test_prediction = final_model.predict(X_test)
    metrics = pd.concat([
        validation_metrics,
        pd.DataFrame([
            score_predictions(MODEL_NAME, "train", y_train, train_prediction),
            score_predictions(MODEL_NAME, "test", y_test, test_prediction),
            score_predictions("Persistence", "test", y_test, X_test["lag_1"]),
        ]),
    ], ignore_index=True)
    print(metrics.to_string(index=False))
    print("Negative test predictions:", int((test_prediction < 0).sum()))

# %% [markdown]
# ## 9. Inspect the learned coefficients
# Each coefficient describes an input's contribution while holding other inputs
# constant. Correlated inputs make individual coefficients harder to interpret.
# We export coefficients on both standardized and original input scales.

# %%
if RUN_TRAINING:
    regressor = final_model.named_steps["regressor"]
    scaler = final_model.named_steps["scaler"]
    raw_coefficients = regressor.coef_ / scaler.scale_
    raw_intercept = float(regressor.intercept_ - (raw_coefficients * scaler.mean_).sum())
    feature_effects = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient_standardized_input": regressor.coef_,
        "coefficient_original_input": raw_coefficients,
    })
    print(feature_effects.to_string(index=False))
    print("Intercept on original input scale:", raw_intercept)

# %% [markdown]
# ## 10. Save the trained pipeline and evidence
# The saved pipeline includes its fitted scaler. Later, give it the same 15
# unscaled feature columns in the same order and call `.predict(new_features)`.
# Training creates `outputs/model.joblib`, metrics, predictions and a chart.

# %%
if RUN_TRAINING:
    save_results(
        OUTPUT_DIR, final_model, metrics,
        validation_metrics.loc[validation_metrics["model"] == MODEL_NAME],
        prediction_table(y_val, validation_prediction, X_val["lag_1"]),
        prediction_table(y_test, test_prediction, X_test["lag_1"]),
        feature_effects, ARTIFACTS_DIR,
        settings={
            "algorithm": MODEL_NAME,
            "fit_intercept": True,
            "scaling": "StandardScaler fitted inside the pipeline",
            "validation_fit_predict_seconds": validation_seconds,
            "final_fit_seconds": final_fit_seconds,
            "intercept_original_input": raw_intercept,
        },
    )

# %% [markdown]
# ## 11. Understand the scope of the results
# These checks preserve your team's existing preprocessing. The following
# assumptions matter when explaining results or choosing a deployed model.

# %%
for limitation in LIMITATIONS:
    print("-", limitation)
