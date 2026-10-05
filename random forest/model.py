# %% [markdown]
# # Random Forest Regression: Steel Industry Energy Consumption
# Predict `Usage_kWh` for one 15-minute interval by combining decision trees.
# This notebook starts in **check-only mode**. Read the code first.
# Training will happen later when `RUN_TRAINING` is True.

# %% [markdown]
# ## 1. Import the tools
# pandas handles tables. scikit-learn provides the forest and settings search.
# Both models use the same shared input checks, dates and scoring functions.

# %%
from pathlib import Path
import os
import sys
from time import perf_counter

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

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
# ## 2. Set paths and the training switch
# Leave `RUN_TRAINING = False` while learning the code. Later, set it to True
# and run from the beginning, or use `model.py --train` from the terminal.
# A fixed random seed makes comparisons repeatable in the same environment.

# %%
RUN_TRAINING = False
ARTIFACTS_DIR = Path(os.environ.get("FDM_ARTIFACTS", str(ROOT.parent / "FDM" / "artifacts")))
OUTPUT_DIR = ROOT / "random forest" / "outputs"
MODEL_NAME = "Random Forest Regression"
RANDOM_STATE = 42

if "__file__" in globals() and __name__ == "__main__":
    args = command_line_settings(ARTIFACTS_DIR)
    RUN_TRAINING = args.train
    ARTIFACTS_DIR = args.artifacts

# %% [markdown]
# ## 3. Load the same prepared inputs and answers
# The timestamp is the row index. X contains input features; y contains the
# actual usage we want to learn to predict. The saved CSV inputs are unscaled.

# %%
X_train, X_test, y_train, y_test = load_data(ARTIFACTS_DIR)
print(X_train.head(3).to_string())
print(y_train.head(3).to_string())

# %% [markdown]
# ## 4. Use the same chronological validation split
# January-August: fitting. September-October: validation and settings selection.
# November onward: final testing. The dates match the Linear Regression model.

# %%
X_fit, X_val, y_fit, y_val = split_for_validation(X_train, y_train)
describe_data(X_train, X_test, X_fit, X_val)

# %% [markdown]
# ## 5. Build the forest (no learning yet)
# `n_estimators=200` means 200 trees. Their predictions are averaged.
# `max_depth` limits tree depth; `min_samples_leaf` limits tiny leaf groups.
# `max_features` controls the fraction of candidate inputs at each split.
# `bootstrap=True` gives trees samples drawn with replacement.
# Scaling is unnecessary for the forest's threshold-based splits.
# `n_jobs=2` allows two workers, keeping resource use modest.

# %%
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    max_features=1.0,
    bootstrap=True,
    random_state=RANDOM_STATE,
    n_jobs=2,
)
parameter_grid = {
    "max_depth": [None, 16],
    "min_samples_leaf": [1, 4],
    "max_features": [1.0, 0.7],
}
print(model)
print("Settings combinations for later training:", len(ParameterGrid(parameter_grid)))
if not RUN_TRAINING:
    print("CHECK COMPLETE. The model is built but untrained. Training cells below are skipped.")

# %% [markdown]
# ## 6. Train eight candidates and choose using validation RMSE
# Each candidate learns from exactly the same fitting rows. Only its settings
# change. The grid includes the starting settings above. Validation RMSE decides
# the winner; the test scores play no role. This is a small search, not proof
# that the settings are globally optimal.

# %%
if RUN_TRAINING:
    trials = []
    best_model = None
    best_params = None
    best_rmse = float("inf")
    for trial_number, params in enumerate(ParameterGrid(parameter_grid), start=1):
        candidate = clone(model).set_params(**params)
        started = perf_counter()
        candidate.fit(X_fit, y_fit)
        candidate_prediction = candidate.predict(X_val)
        elapsed = perf_counter() - started
        scores = score_predictions(MODEL_NAME, "validation", y_val, candidate_prediction)
        trials.append({"trial": trial_number, **params, **scores, "fit_predict_seconds": elapsed})
        print(f"Trial {trial_number}/8: RMSE={scores['RMSE_kWh']:.4f} kWh; {params}", flush=True)
        if scores["RMSE_kWh"] < best_rmse:
            best_rmse = scores["RMSE_kWh"]
            best_params = params.copy()
            best_model = candidate
    validation_trials = pd.DataFrame(trials).sort_values("RMSE_kWh", kind="stable")
    validation_prediction = best_model.predict(X_val)
    validation_metrics = pd.DataFrame([
        score_predictions(MODEL_NAME, "validation", y_val, validation_prediction),
        score_predictions("Persistence", "validation", y_val, X_val["lag_1"]),
    ])
    print("Selected settings:", best_params)
    print(validation_metrics.to_string(index=False))

# %% [markdown]
# ## 7. Refit the selected settings on all training rows
# `clone` creates a fresh forest with the selected settings. It now learns
# from all January-October records. No test targets are passed to `.fit()`.

# %%
if RUN_TRAINING:
    final_model = clone(best_model)
    started = perf_counter()
    final_model.fit(X_train, y_train)
    final_fit_seconds = perf_counter() - started

# %% [markdown]
# ## 8. Evaluate using regression metrics
# MAE is the average absolute error in kWh. RMSE penalizes larger mistakes more.
# Lower MAE/RMSE and higher R2 are better. R2 is not percentage accuracy.
# The persistence baseline predicts the previous interval's actual usage.
# Compare model selection using the validation rows; report test results after
# settings are fixed. A much better training score can suggest overfitting.

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

# %% [markdown]
# ## 9. Inspect which inputs the forest used
# Feature importance measures how much inputs helped reduce error in the trees'
# training splits. It is not evidence of cause and effect. Correlated features
# can share importance, and this measure can favour continuous inputs.

# %%
if RUN_TRAINING:
    feature_effects = pd.DataFrame({
        "feature": X_train.columns,
        "impurity_importance": final_model.feature_importances_,
    }).sort_values("impurity_importance", ascending=False)
    print(feature_effects.to_string(index=False))

# %% [markdown]
# ## 10. Save the trained forest and evidence
# The `outputs` folder is created only after training. It will contain a saved
# model, the eight validation trials, regression scores, predictions and a chart.
# Future `.predict(new_features)` calls need the same 15 unscaled feature columns.

# %%
if RUN_TRAINING:
    save_results(
        OUTPUT_DIR, final_model, metrics, validation_trials,
        prediction_table(y_val, validation_prediction, X_val["lag_1"]),
        prediction_table(y_test, test_prediction, X_test["lag_1"]),
        feature_effects, ARTIFACTS_DIR,
        settings={
            "algorithm": MODEL_NAME,
            "selected_parameters": best_params,
            "n_estimators": 200,
            "random_state": RANDOM_STATE,
            "search_candidates": len(trials),
            "search_fit_predict_seconds": float(validation_trials["fit_predict_seconds"].sum()),
            "final_fit_seconds": final_fit_seconds,
        },
    )

# %% [markdown]
# ## 11. Understand the scope of the results
# These assumptions come from the supplied preprocessing and apply to both models.

# %%
for limitation in LIMITATIONS:
    print("-", limitation)
