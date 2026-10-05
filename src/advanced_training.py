"""Extended training of only LinearRegression and RandomForestRegressor."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterSampler, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.optimized_regression import (
    ForecastFeatures, SMOOTH_FEATURES, checked_features, feature_effects,
    make_estimator as previous_estimator,
)
from src.regression_common import (
    FEATURES, LIMITATIONS, load_data, prediction_table, save_results, score_predictions,
)
from src.show_model_results import high_load_mae


class ScheduleFeatures(ForecastFeatures):
    """Extend existing causal features with schedule and recent-change interactions."""

    def __init__(self, mode="context", use_load=True, schedule="none"):
        super().__init__(mode=mode, use_load=use_load)
        self.schedule = schedule

    def _transform(self, X):
        out = super()._transform(X)
        extra = {
            "upward_change": X["diff_1"].clip(lower=0),
            "downward_change": X["diff_1"].clip(upper=0),
            "daily_lag_deviation": X["lag_1"] - X["lag_96"],
            "weekly_lag_deviation": X["lag_1"] - X["lag_672"],
            "variation_relative_to_recent": X["roll_std_4"] / (1 + X["roll_mean_4"].clip(lower=0)),
        }
        angle = np.mod(np.arctan2(X["tod_sin"], X["tod_cos"]), 2 * np.pi)
        if self.schedule == "weekend":
            slot = np.rint(angle * 96 / (2 * np.pi)).astype(int) % 96
            for value in range(1, 96):
                active = ((slot == value) & (X["is_weekend"] == 1)).astype(float)
                extra[f"weekend_slot_{value}"] = active
                extra[f"weekend_slot_{value}_x_lag_1"] = active * X["lag_1"]
        elif self.schedule == "harmonics":
            for harmonic in [2, 3, 4]:
                for name, wave in [("sin", np.sin(harmonic * angle)), ("cos", np.cos(harmonic * angle))]:
                    extra[f"tod_{harmonic}_{name}"] = wave
                    for day_feature in ["dow_sin", "dow_cos", "is_weekend"]:
                        interaction = wave * X[day_feature]
                        extra[f"tod_{harmonic}_{name}_x_{day_feature}"] = interaction
                        extra[f"tod_{harmonic}_{name}_x_{day_feature}_x_lag_1"] = interaction * X["lag_1"]
        elif self.schedule != "none":
            raise ValueError(f"Unknown schedule: {self.schedule}")
        return pd.concat([out, pd.DataFrame(extra, index=X.index)], axis=1)


class AdvancedEnergyRegressor(RegressorMixin, BaseEstimator):
    """Learn usage or its change, with optional fit-only example weighting."""

    def __init__(self, estimator, target_mode="level", high_weight=1.0, half_life_days=None):
        self.estimator = estimator
        self.target_mode = target_mode
        self.high_weight = high_weight
        self.half_life_days = half_life_days

    def fit(self, X, y):
        checked_features(X)
        values = np.asarray(y, dtype=float)
        if values.ndim != 1 or len(values) != len(X) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Expected one finite nonnegative energy target per input row.")
        if isinstance(y, pd.Series) and not y.index.equals(X.index):
            raise ValueError("Input and target timestamps do not align.")
        if self.high_weight <= 0:
            raise ValueError("Training weights must be positive.")
        weights = np.where(values >= 50, self.high_weight, 1.0)
        if self.half_life_days is not None:
            if self.half_life_days <= 0 or not isinstance(X.index, pd.DatetimeIndex):
                raise ValueError("Recency weighting requires datetime rows and a positive half-life.")
            age_days = np.asarray((X.index.max() - X.index).total_seconds()) / 86400
            weights *= np.exp2(-age_days / self.half_life_days)
        weights /= weights.mean()
        if self.target_mode == "delta":
            target = values - X["lag_1"].to_numpy()
        elif self.target_mode == "level":
            target = values
        else:
            raise ValueError(f"Unknown target mode: {self.target_mode}")
        self.estimator_ = clone(self.estimator).fit(X, target, regressor__sample_weight=weights)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_raw(self, X):
        check_is_fitted(self, "estimator_")
        predicted = self.estimator_.predict(checked_features(X))
        if self.target_mode == "delta":
            predicted = predicted + X["lag_1"].to_numpy()
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite model predictions.")
        return predicted

    def predict(self, X):
        return np.maximum(0, self.predict_raw(X))


def make_model(kind, specification, jobs=4):
    spec = specification.copy()
    if "previous" in spec:
        return previous_estimator(kind, spec["previous"], jobs)
    features = ScheduleFeatures(
        mode="time96" if kind == "linear" else "context",
        schedule=spec.pop("schedule", "none"), use_load=spec.pop("use_load", True),
    )
    target_mode = spec.pop("target_mode", "level")
    high_weight = spec.pop("high_weight", 1.0)
    half_life = spec.pop("half_life_days", None)
    steps = [("features", features)]
    if kind == "linear":
        steps.append(("basis", ColumnTransformer([
            ("smooth", SplineTransformer(
                n_knots=spec.pop("knots", 5), degree=spec.pop("degree", 3),
                knots=spec.pop("knot_spacing", "quantile"), include_bias=False,
                extrapolation="linear",
            ), SMOOTH_FEATURES),
        ], remainder="passthrough", verbose_feature_names_out=False)))
        steps.extend([("scaler", StandardScaler()), ("regressor", LinearRegression())])
        if spec:
            raise ValueError(f"Unused linear settings: {spec}")
    elif kind == "forest":
        steps.append(("regressor", RandomForestRegressor(random_state=42, n_jobs=jobs, **spec)))
    else:
        raise ValueError(f"Unknown kind: {kind}")
    return AdvancedEnergyRegressor(Pipeline(steps), target_mode, high_weight, half_life)


def candidate_specs(kind, incumbent):
    specs = [{"previous": incumbent}]
    if kind == "linear":
        for schedule in ["none", "weekend", "harmonics"]:
            for knots in [3, 5, 8]:
                for use_load in [True, False]:
                    specs.append({"schedule": schedule, "knots": knots, "use_load": use_load})
        for schedule in ["none", "weekend", "harmonics"]:
            for extra in [
                {"high_weight": 1.3}, {"high_weight": 2.0},
                {"half_life_days": 60}, {"half_life_days": 120},
                {"degree": 2}, {"knot_spacing": "uniform"},
            ]:
                specs.append({"schedule": schedule, "knots": 5, **extra})
    else:
        space = {
            "schedule": ["none", "harmonics"], "use_load": [True, False],
            "target_mode": ["level", "delta"], "high_weight": [1.0, 1.3, 2.0],
            "half_life_days": [None, 90], "max_depth": [8, 12, 18, None],
            "min_samples_leaf": [1, 2, 4, 8], "min_samples_split": [2, 10, 20],
            "max_features": [0.35, 0.6, 1.0], "max_samples": [0.7, None],
        }
        specs.extend({**p, "n_estimators": 128} for p in ParameterSampler(space, n_iter=48, random_state=42))
        for mode in ["level", "delta"]:
            for use_load in [True, False]:
                specs.append({"target_mode": mode, "use_load": use_load, "max_depth": 12,
                              "min_samples_split": 10, "max_features": 0.4, "n_estimators": 128})
    return specs


def extended_scores(name, split, actual, predicted):
    return {**score_predictions(name, split, actual, predicted),
            "HighLoad_MAE": high_load_mae(actual, predicted),
            "high_load_rows": int((np.asarray(actual) >= 50).sum())}


def search(kind, X, y, incumbent, directory, jobs=4):
    """Screen on the last two folds, then choose finalists by full five-fold mean RMSE."""
    folds = list(TimeSeriesSplit(n_splits=5).split(X))
    specs = candidate_specs(kind, incumbent)
    records, fold_records, cache = [], [], {}
    names = ["MAE_kWh", "RMSE_kWh", "R2", "HighLoad_MAE"]
    fold_table = pd.DataFrame([
        {"fold": i + 1, "fit_rows": len(tr), "validation_rows": len(va),
         "fit_start": str(X.index[tr[0]]), "fit_end": str(X.index[tr[-1]]),
         "validation_start": str(X.index[va[0]]), "validation_end": str(X.index[va[-1]])}
        for i, (tr, va) in enumerate(folds)
    ])
    fold_table.to_csv(directory / "fold_boundaries.csv", index=False)

    def evaluate(spec, fold_ids, phase):
        key = json.dumps(spec, sort_keys=True)
        scores = []
        for fold_id in fold_ids:
            cache_key = (key, fold_id)
            if cache_key not in cache:
                train, valid = folds[fold_id]
                started = perf_counter()
                model = make_model(kind, spec, jobs).fit(X.iloc[train], y.iloc[train])
                predicted = model.predict(X.iloc[valid])
                score = extended_scores(kind, f"fold_{fold_id + 1}", y.iloc[valid], predicted)
                score["fit_predict_seconds"] = perf_counter() - started
                cache[cache_key] = (score, predicted)
                fold_records.append({"specification": key, "fold": fold_id + 1, **score})
            scores.append(cache[cache_key][0])
        row = {"trial": len(records) + 1, "phase": phase, "n_folds": len(fold_ids), "specification": key,
               **{name: float(np.mean([s[name] for s in scores])) for name in names},
               "RMSE_std": float(np.std([s["RMSE_kWh"] for s in scores])),
               "fit_predict_seconds": sum(s["fit_predict_seconds"] for s in scores)}
        records.append(row)
        pd.DataFrame(records).to_csv(directory / "search_checkpoint.csv", index=False)
        pd.DataFrame(fold_records).to_csv(directory / "cv_fold_trials.csv", index=False)
        print(f"{kind} {phase} {row['trial']}: {len(fold_ids)}-fold RMSE={row['RMSE_kWh']:.5f}; high-load MAE={row['HighLoad_MAE']:.5f}", flush=True)
        return row

    for spec in specs:
        evaluate(spec, [3, 4], "screen")
    shortlist = sorted(records, key=lambda row: row["RMSE_kWh"])[:8 if kind == "linear" else 6]
    finalists = [{"previous": incumbent}]
    for row in shortlist:
        spec = json.loads(row["specification"])
        if "previous" not in spec:
            if kind == "forest":
                spec["n_estimators"] = 400
            finalists.append(spec)
    for spec in finalists:
        evaluate(spec, list(range(5)), "full_cv")
    if kind == "forest":
        ranked = sorted([r for r in records if r["phase"] == "full_cv" and "previous" not in json.loads(r["specification"])], key=lambda r: r["RMSE_kWh"])
        for row in ranked[:2]:
            evaluate({**json.loads(row["specification"]), "n_estimators": 1000}, list(range(5)), "full_cv")
    full = sorted([r for r in records if r["phase"] == "full_cv"], key=lambda r: r["RMSE_kWh"])
    selected = json.loads(full[0]["specification"])
    chosen_key = json.dumps(selected, sort_keys=True)
    prediction_frames = []
    for fold_id, (_, valid) in enumerate(folds):
        frame = prediction_table(y.iloc[valid], cache[(chosen_key, fold_id)][1], X.iloc[valid]["lag_1"])
        frame["fold"] = fold_id + 1
        prediction_frames.append(frame)
    return selected, pd.DataFrame(records), pd.DataFrame(fold_records), pd.concat(prediction_frames), fold_table


def missing_reading_sensitivity(test_predictions):
    """Exclude the known interpolated test answer and features influenced by it."""
    bad = pd.Timestamp("2018-11-08 00:00")
    affected = {bad + pd.Timedelta(minutes=15 * offset) for offset in [*range(97), 672]}
    clean = test_predictions.loc[~test_predictions.index.isin(affected)]
    return extended_scores("Selected configuration", "exclude_interpolation_influence", clean.actual_kWh, clean.predicted_kWh)


def run(kind, artifacts, directory, jobs=4):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    folder = "linear regression" if kind == "linear" else "random forest"
    title = "Linear Regression" if kind == "linear" else "Random Forest Regression"
    previous = json.loads((ROOT / folder / "outputs_optimized" / "selected_configuration.json").read_text())["selected_specification"]
    X_train, X_test, y_train, y_test = load_data(artifacts)
    specs = candidate_specs(kind, previous)
    protocol = {
        "algorithm": title, "screen_candidates": len(specs), "screen_folds": [4, 5],
        "selection": "mean of five TimeSeriesSplit fold RMSE values; lowest wins",
        "n_splits": 5, "random_state": 42, "high_load_definition": "actual Usage_kWh >= 50",
        "test_used_for_selection": False, "test_previously_seen_in_project": True,
        "input_feature_contract": FEATURES,
        "search_space": specs,
    }
    (directory / "search_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"{title}: {len(specs)} screening candidates, followed by five-fold finalist evaluation.", flush=True)
    started = perf_counter()
    with threadpool_limits(limits=jobs):
        selected, trials, folds, oof, boundaries = search(kind, X_train, y_train, previous, directory, jobs)
        full = trials.loc[trials.phase == "full_cv"].sort_values("RMSE_kWh", kind="stable")
        best = full.iloc[0]
        selection = {"algorithm": title, "selected_specification": selected,
                     "selection_metric": protocol["selection"], "cv_RMSE_kWh": float(best.RMSE_kWh),
                     "cv_HighLoad_MAE": float(best.HighLoad_MAE), "test_used_for_selection": False}
        (directory / "selected_configuration.json").write_text(json.dumps(selection, indent=2) + "\n")
        print("SELECTED BEFORE TEST EVALUATION:", json.dumps(selected, sort_keys=True), flush=True)
        fit_started = perf_counter()
        model = make_model(kind, selected, jobs).fit(X_train, y_train)
        fit_seconds = perf_counter() - fit_started
        train_prediction, test_prediction = model.predict(X_train), model.predict(X_test)
        metrics = pd.DataFrame([
            {"model": title, "split": "cv_mean", "n_rows": len(oof),
             **{name: float(best[name]) for name in ["MAE_kWh", "RMSE_kWh", "R2", "HighLoad_MAE"]}},
            extended_scores(title, "cv_pooled", oof.actual_kWh, oof.predicted_kWh),
            extended_scores("Persistence", "cv_pooled", oof.actual_kWh, oof.persistence_kWh),
            extended_scores(title, "train", y_train, train_prediction),
            extended_scores(title, "test", y_test, test_prediction),
            extended_scores("Persistence", "test", y_test, X_test.lag_1),
        ])
        effects = feature_effects(model)
        test_frame = prediction_table(y_test, test_prediction, X_test.lag_1)
        run_limitations = list(LIMITATIONS)
        if not model.estimator_.named_steps["features"].use_load:
            run_limitations[1] = (
                "The selected transformer ignores load_level. It remains in the 15-column "
                "input schema, but no upcoming operating-load category is required."
            )
        settings = {
            **selection, "search_candidates": len(trials), "screen_candidates": len(specs),
            "full_cv_candidates": len(full), "raw_input_features": len(FEATURES),
            "transformed_features": len(effects), "final_fit_rows": len(X_train),
            "final_fit_seconds": fit_seconds, "total_seconds_before_export": perf_counter() - started,
            "validation_start": boundaries.iloc[0].validation_start,
            "postprocessing": "Predictions floored at zero; included in saved model",
            "model_compression": 3,
            "target_mode": selected.get("target_mode", "level"),
            "validation_note": "Non-nested CV used for tuning, not an unbiased independent estimate.",
            "limitations": run_limitations,
        }
        save_results(directory, model, metrics, trials, oof, test_frame, effects, artifacts, settings)
        model_path = directory / "model.joblib"
        reloaded = joblib.load(model_path)
        np.testing.assert_allclose(reloaded.predict(X_test), test_prediction, rtol=1e-10, atol=1e-8)
        check_rows = X_test.iloc[:96]
        pred_started = perf_counter()
        reloaded.predict(check_rows)
        prediction_seconds = perf_counter() - pred_started
        verification = {
            "reload_predictions_match": True, "fitted_on_rows": len(X_train),
            "test_rows": len(X_test), "prediction_seconds_96_rows": prediction_seconds,
            "model_size_bytes": model_path.stat().st_size,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (directory / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        pd.DataFrame([missing_reading_sensitivity(test_frame)]).to_csv(directory / "interpolation_sensitivity.csv", index=False)
        group_rows = []
        for period in y_test.index.to_period("M").unique():
            mask = y_test.index.to_period("M") == period
            group_rows.append(extended_scores(title, str(period), y_test.loc[mask], test_prediction[mask]))
        pd.DataFrame(group_rows).to_csv(directory / "test_monthly_metrics.csv", index=False)
        # Settings are now fixed. This estimates dependence on the assumed-known load category.
        no_load = clone(model).set_params(estimator__features__use_load=False)
        last_tr, last_va = list(TimeSeriesSplit(n_splits=5).split(X_train))[-1]
        no_load.fit(X_train.iloc[last_tr], y_train.iloc[last_tr])
        load_sensitivity = extended_scores("Selected settings without load_level", "last_cv_fold", y_train.iloc[last_va], no_load.predict(X_train.iloc[last_va]))
        pd.DataFrame([load_sensitivity]).to_csv(directory / "load_level_ablation.csv", index=False)
    print(metrics.to_string(index=False), flush=True)
    print(f"Advanced {kind} finished and reload verified in {perf_counter() - started:.1f} seconds.", flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["linear", "forest"], required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--artifacts", type=Path, default=ROOT.parent / "FDM" / "artifacts")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive.")
    folder = "linear regression" if args.model == "linear" else "random forest"
    if args.train:
        run(args.model, args.artifacts, ROOT / folder / "outputs_advanced", args.jobs)
    else:
        print("Add --train to run the advanced experiment. Existing artifacts are preserved.")


if __name__ == "__main__":
    # Keep pickle class paths importable when this file is launched directly.
    from src.advanced_training import main as entry_main

    entry_main()
