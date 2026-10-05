"""Reproducible regression optimization using only chronological validation scores."""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

from src.regression_common import (
    FEATURES, LIMITATIONS, load_data, prediction_table, save_results,
    score_predictions, split_for_validation,
)

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
SMOOTH_FEATURES = [
    "lag_1", "lag_4", "lag_96", "lag_672", "roll_mean_4", "roll_std_4",
    "roll_mean_96", "roll_max_96", "diff_1",
]


def checked_features(X):
    if not isinstance(X, pd.DataFrame) or list(X.columns) != FEATURES:
        raise ValueError("Supply the 15 named, unscaled feature columns in their saved order.")
    if X.empty or not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Features must be nonempty, numeric and finite.")
    return X


class ForecastFeatures(TransformerMixin, BaseEstimator):
    """Derive inputs from a row's existing information, without accessing y or other rows."""

    def __init__(self, mode="original", use_load=True):
        self.mode = mode
        self.use_load = use_load

    def fit(self, X, y=None):
        checked_features(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.output_names_ = np.asarray(self._transform(X.iloc[:1]).columns, dtype=object)
        return self

    def _transform(self, X):
        out = X.copy()
        if not self.use_load:
            out = out.drop(columns="load_level")
        if self.mode == "original":
            return out
        if self.mode not in {"context", "time24", "time48", "time96"}:
            raise ValueError(f"Unknown feature mode: {self.mode}")
        angle = np.mod(np.arctan2(X["tod_sin"], X["tod_cos"]), 2 * np.pi)
        slot = np.rint(angle * 96 / (2 * np.pi)).astype(int) % 96
        day_angle = np.mod(np.arctan2(X["dow_sin"], X["dow_cos"]), 2 * np.pi)
        weekday = np.rint(day_angle * 7 / (2 * np.pi)).astype(int) % 7
        if self.mode == "context":
            out["quarter_hour"] = slot
            out["weekday"] = weekday
            out["lag_2_reconstructed"] = X["lag_1"] - X["diff_1"]
            out["lag_3_reconstructed"] = (
                4 * X["roll_mean_4"] - X["lag_1"] - out["lag_2_reconstructed"] - X["lag_4"]
            )
        additions = {
            "recent_deviation": X["lag_1"] - X["roll_mean_4"],
            "daily_deviation": X["lag_1"] - X["roll_mean_96"],
            "relative_recent_usage": X["lag_1"] / (1 + X["roll_mean_4"].clip(lower=0)),
            "relative_daily_usage": X["lag_1"] / (1 + X["roll_mean_96"].clip(lower=0)),
            "relative_change": X["diff_1"] / (1 + X["lag_1"].clip(lower=0)),
        }
        if self.mode.startswith("time"):
            bins = int(self.mode[4:])
            bucket = slot // (96 // bins)
            # Drop a reference bucket: the intercept and original lags cover it.
            for value in range(1, bins):
                indicator = (bucket == value).astype(float)
                additions[f"time_{value:02d}"] = indicator
                for name in ["lag_1", "roll_mean_4", "diff_1"]:
                    additions[f"time_{value:02d}_x_{name}"] = indicator * X[name]
            for value in range(1, 7):
                indicator = (weekday == value).astype(float)
                additions[f"weekday_{value}"] = indicator
                additions[f"weekday_{value}_x_lag_1"] = indicator * X["lag_1"]
            if self.use_load:
                for value in [1, 2]:
                    indicator = (X["load_level"] == value).astype(float)
                    additions[f"load_{value}"] = indicator
                    additions[f"load_{value}_x_lag_1"] = indicator * X["lag_1"]
        return pd.concat([out, pd.DataFrame(additions, index=X.index)], axis=1)

    def transform(self, X):
        check_is_fitted(self, "output_names_")
        return self._transform(checked_features(X))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "output_names_")
        return self.output_names_.copy()


class EnergyRegressor(RegressorMixin, BaseEstimator):
    """Save preprocessing, estimator and the fixed nonnegative prediction rule together."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        checked_features(X)
        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or len(values) != len(X) or not np.isfinite(values).all():
            raise ValueError("Supply one finite target for each input row.")
        if (values < 0).any():
            raise ValueError("Energy targets must be nonnegative.")
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_raw(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(checked_features(X))

    def predict(self, X):
        # Negative energy consumption is outside this dataset's target domain.
        return np.maximum(0.0, self.predict_raw(X))


def make_estimator(kind, specification, jobs=4):
    spec = specification.copy()
    mode = spec.pop("feature_mode", "original")
    steps = [("features", ForecastFeatures(mode=mode))]
    if kind == "linear":
        basis = spec.pop("basis", "linear")
        if basis in {"interactions", "quadratic"}:
            steps.append(("basis", PolynomialFeatures(
                degree=2, include_bias=False, interaction_only=basis == "interactions",
            )))
        elif basis == "spline":
            steps.append(("basis", ColumnTransformer([
                ("smooth", SplineTransformer(
                    n_knots=spec.pop("knots"), degree=3, knots="quantile",
                    include_bias=False, extrapolation="linear",
                ), SMOOTH_FEATURES),
            ], remainder="passthrough", verbose_feature_names_out=False)))
        elif basis != "linear":
            raise ValueError(f"Unknown basis: {basis}")
        if spec:
            raise ValueError(f"Unused linear settings: {spec}")
        steps.extend([("scaler", StandardScaler()), ("regressor", LinearRegression())])
    elif kind == "forest":
        steps.append(("regressor", RandomForestRegressor(
            random_state=SEED, n_jobs=jobs, **spec,
        )))
    else:
        raise ValueError(f"Unknown algorithm: {kind}")
    return EnergyRegressor(Pipeline(steps))


def linear_candidates():
    candidates = [
        {"feature_mode": "original", "basis": basis}
        for basis in ["linear", "interactions", "quadratic"]
    ]
    for mode in ["original", "time24", "time48", "time96"]:
        if mode != "original":
            candidates.append({"feature_mode": mode, "basis": "linear"})
        for knots in [3, 5, 7]:
            candidates.append({"feature_mode": mode, "basis": "spline", "knots": knots})
    return candidates


def forest_candidates():
    incumbent = {
        "feature_mode": "original", "n_estimators": 200, "max_depth": 16,
        "max_features": 0.7, "min_samples_leaf": 4, "min_samples_split": 2,
        "max_samples": None,
    }
    candidates = [incumbent, {**incumbent, "feature_mode": "context"}]
    space = {
        "feature_mode": ["original", "context"],
        "max_depth": [12, 16, 22, 30, None],
        "max_features": [0.4, 0.6, 0.8, 1.0],
        "min_samples_leaf": [1, 2, 4, 8],
        "min_samples_split": [2, 5, 10],
        "max_samples": [0.65, 0.85, None],
    }
    candidates.extend({**p, "n_estimators": 160} for p in ParameterSampler(
        space, n_iter=36, random_state=SEED,
    ))
    return candidates


def search(kind, X_fit, y_fit, X_val, y_val, output_dir, jobs=4):
    """No test features or test targets enter candidate selection."""
    specs = linear_candidates() if kind == "linear" else forest_candidates()
    trials, best, best_spec = [], None, None
    best_rmse = float("inf")

    def trial(spec, stage):
        nonlocal best, best_spec, best_rmse
        started = perf_counter()
        candidate = make_estimator(kind, spec, jobs).fit(X_fit, y_fit)
        raw = candidate.predict_raw(X_val)
        scores = score_predictions(kind, "validation", y_val, np.maximum(0, raw))
        record = {
            "trial": len(trials) + 1, "stage": stage, "specification": json.dumps(spec, sort_keys=True),
            **scores, "raw_RMSE_kWh": score_predictions(kind, "validation", y_val, raw)["RMSE_kWh"],
            "negative_raw_predictions": int((raw < 0).sum()),
            "fit_predict_seconds": perf_counter() - started,
        }
        trials.append(record)
        pd.DataFrame(trials).to_csv(output_dir / "search_checkpoint.csv", index=False)
        print(f"{kind} trial {len(trials)} ({stage}): validation RMSE={scores['RMSE_kWh']:.5f}; {spec}", flush=True)
        if scores["RMSE_kWh"] < best_rmse:
            best, best_spec, best_rmse = candidate, spec.copy(), scores["RMSE_kWh"]

    for spec in specs:
        trial(spec, "initial")
    if kind == "forest":
        shortlist = sorted(trials, key=lambda row: row["RMSE_kWh"])[:4]
        for row in shortlist:
            spec = json.loads(row["specification"])
            for trees in [400, 800]:
                trial({**spec, "n_estimators": trees}, "more_trees")
    table = pd.DataFrame(trials).sort_values("RMSE_kWh", kind="stable")
    return best, best_spec, table


def walk_forward_diagnostics(estimator, X_train, y_train, validation_scores):
    rows = []
    for start, end in [("2018-05-01", "2018-07-01"), ("2018-07-01", "2018-09-01")]:
        fit = X_train.index < pd.Timestamp(start)
        valid = (X_train.index >= pd.Timestamp(start)) & (X_train.index < pd.Timestamp(end))
        model = clone(estimator).fit(X_train.loc[fit], y_train.loc[fit])
        scores = score_predictions("Selected configuration", start[:7], y_train.loc[valid], model.predict(X_train.loc[valid]))
        rows.append({**scores, "fit_rows": int(fit.sum()), "validation_end_exclusive": end})
        baseline = score_predictions("Persistence", start[:7], y_train.loc[valid], X_train.loc[valid, "lag_1"])
        rows.append({**baseline, "fit_rows": int(fit.sum()), "validation_end_exclusive": end})
        print(f"Walk-forward diagnostic {start} to {end}: RMSE={scores['RMSE_kWh']:.5f}", flush=True)
    rows.extend({**row, "fit_rows": int((X_train.index < pd.Timestamp("2018-09-01")).sum()),
                 "validation_end_exclusive": "2018-11-01"} for row in validation_scores.to_dict("records"))
    return pd.DataFrame(rows)


def feature_effects(model):
    pipeline = model.estimator_
    names = pipeline[:-1].get_feature_names_out()
    regressor = pipeline.named_steps["regressor"]
    if isinstance(regressor, RandomForestRegressor):
        return pd.DataFrame({"feature": names, "impurity_importance": regressor.feature_importances_}).sort_values(
            "impurity_importance", ascending=False,
        )
    scaler = pipeline.named_steps["scaler"]
    return pd.DataFrame({
        "feature": names,
        "coefficient_standardized_basis": regressor.coef_,
        "coefficient_unscaled_basis": regressor.coef_ / scaler.scale_,
    })


def run(kind, artifacts_dir, output_dir, jobs=4):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_data(artifacts_dir)
    X_fit, X_val, y_fit, y_val = split_for_validation(X_train, y_train)
    title = "Linear Regression" if kind == "linear" else "Random Forest Regression"
    print(f"Optimizing {title}: {len(X_fit):,} fitting rows; {len(X_val):,} validation rows.", flush=True)
    started = perf_counter()
    # Limit BLAS as well as forest workers so the laptop remains usable.
    with threadpool_limits(limits=jobs):
        best, specification, trials = search(kind, X_fit, y_fit, X_val, y_val, output_dir, jobs)
        validation_prediction = best.predict(X_val)
        validation_metrics = pd.DataFrame([
            score_predictions(title, "validation", y_val, validation_prediction),
            score_predictions("Persistence", "validation", y_val, X_val["lag_1"]),
        ])
        selection = {
            "algorithm": title, "learner": "LinearRegression" if kind == "linear" else "RandomForestRegressor",
            "selected_specification": specification,
            "selection_metric": "September-October validation RMSE_kWh",
            "best_validation_RMSE_kWh": float(trials.iloc[0]["RMSE_kWh"]),
            "test_used_for_selection": False, "search_candidates": len(trials),
        }
        (output_dir / "selected_configuration.json").write_text(json.dumps(selection, indent=2) + "\n")
        print("Configuration fixed before final test evaluation:", specification, flush=True)
        diagnostics = walk_forward_diagnostics(best, X_train, y_train, validation_metrics)
        diagnostics.to_csv(output_dir / "walk_forward_diagnostics.csv", index=False)

        ablated = clone(best).set_params(estimator__features__use_load=False)
        ablated.fit(X_fit, y_fit)
        ablation = pd.DataFrame([
            score_predictions("Selected, includes load_level", "validation", y_val, validation_prediction),
            score_predictions("Selected, without load_level", "validation", y_val, ablated.predict(X_val)),
        ])
        ablation.to_csv(output_dir / "load_level_ablation.csv", index=False)
        fit_started = perf_counter()
        final_model = clone(best).fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_started
        train_prediction = final_model.predict(X_train)
        test_prediction = final_model.predict(X_test)
        raw_test = final_model.predict_raw(X_test)
        metrics = pd.concat([validation_metrics, pd.DataFrame([
            score_predictions(title, "train", y_train, train_prediction),
            score_predictions(title, "test", y_test, test_prediction),
            score_predictions("Persistence", "test", y_test, X_test["lag_1"]),
        ])], ignore_index=True)
        print(metrics.to_string(index=False), flush=True)
        effects = feature_effects(final_model)
        transformed_count = len(effects)
        settings = {
            **selection, "postprocessing": "Predictions floored at zero for nonnegative energy consumption",
            "raw_test_metrics": score_predictions(title, "test_unclipped", y_test, raw_test),
            "negative_raw_test_predictions": int((raw_test < 0).sum()),
            "raw_input_features": len(FEATURES), "transformed_features": transformed_count,
            "final_fit_rows": len(X_train), "final_fit_seconds": fit_seconds,
            "total_seconds_before_export": perf_counter() - started, "seed": SEED,
            "walk_forward_note": "Retrospective robustness diagnostics; configuration selected using later Sep-Oct data.",
            "ablation_note": "Diagnostic only; not a replacement selection or proof load category is known in advance.",
        }
        save_results(
            output_dir, final_model, metrics, trials,
            prediction_table(y_val, validation_prediction, X_val["lag_1"]),
            prediction_table(y_test, test_prediction, X_test["lag_1"]),
            effects, artifacts_dir, settings,
        )
        monthly = []
        for period in y_test.index.to_period("M").unique():
            mask = y_test.index.to_period("M") == period
            monthly.append(score_predictions(title, str(period), y_test.loc[mask], test_prediction[mask]))
        pd.DataFrame(monthly).to_csv(output_dir / "test_monthly_metrics.csv", index=False)
    print(f"Optimization and training finished in {perf_counter() - started:.1f} seconds.", flush=True)
    return final_model, metrics


def cli(kind):
    folder = "linear regression" if kind == "linear" else "random forest"
    parser = argparse.ArgumentParser(description=f"Optimize and train {folder} with chronological validation.")
    parser.add_argument("--train", action="store_true", help="Run the search, diagnostics and final training.")
    parser.add_argument("--artifacts", type=Path, default=Path(os.environ.get("FDM_ARTIFACTS", ROOT.parent / "FDM" / "artifacts")))
    parser.add_argument("--output", type=Path, default=ROOT / folder / "outputs_optimized")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if not 1 <= args.jobs <= (os.cpu_count() or 1):
        parser.error("--jobs must be between 1 and the available CPU count.")
    if args.train:
        run(kind, args.artifacts, args.output, args.jobs)
    else:
        load_data(args.artifacts)
        count = len(linear_candidates()) if kind == "linear" else len(forest_candidates()) + 8
        print(f"Check passed. Planned candidates: {count}. Add --train to optimize and save results.")

