"""Check forecast causality, fit isolation and reloadable predictions."""

from io import BytesIO
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.optimized_regression import ForecastFeatures, make_estimator
from src.regression_common import FEATURES


def example_features():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.uniform(1, 20, (160, len(FEATURES))), columns=FEATURES)
    slot = np.arange(len(X)) % 96
    day = np.arange(len(X)) % 7
    X["tod_sin"] = np.sin(2 * np.pi * slot / 96)
    X["tod_cos"] = np.cos(2 * np.pi * slot / 96)
    X["dow_sin"] = np.sin(2 * np.pi * day / 7)
    X["dow_cos"] = np.cos(2 * np.pi * day / 7)
    X["is_weekend"] = (day >= 5).astype(int)
    X["load_level"] = np.arange(len(X)) % 3
    return X


class OptimizationChecks(unittest.TestCase):
    def test_features_are_independent_of_later_rows(self):
        X = example_features()
        for mode in ["original", "context", "time24", "time48", "time96"]:
            with self.subTest(mode=mode):
                transformer = ForecastFeatures(mode).fit(X.iloc[:100])
                batch = transformer.transform(X)
                alone = transformer.transform(X.iloc[[120]])
                pd.testing.assert_frame_equal(batch.iloc[[120]], alone)
                self.assertEqual(len(transformer.get_feature_names_out()), batch.shape[1])

    def test_known_calendar_and_reconstructed_lags(self):
        X = example_features()
        out = ForecastFeatures("context").fit_transform(X)
        np.testing.assert_array_equal(out["quarter_hour"], np.arange(len(X)) % 96)
        np.testing.assert_array_equal(out["weekday"], np.arange(len(X)) % 7)
        np.testing.assert_allclose(out["lag_2_reconstructed"], X["lag_1"] - X["diff_1"])
        np.testing.assert_allclose(
            (X["lag_1"] + out["lag_2_reconstructed"] + out["lag_3_reconstructed"] + X["lag_4"]) / 4,
            X["roll_mean_4"],
        )

    def test_ablation_removes_all_load_information(self):
        X = example_features()
        changed = X.copy()
        changed["load_level"] = 99
        for mode in ["original", "context", "time48"]:
            transformer = ForecastFeatures(mode, use_load=False).fit(X)
            pd.testing.assert_frame_equal(transformer.transform(X), transformer.transform(changed))

    def test_pipeline_scaler_fits_only_the_given_training_rows(self):
        X = example_features()
        y = 2 * X["lag_1"] + 3
        model = make_estimator("linear", {"feature_mode": "original", "basis": "linear"})
        model.fit(X.iloc[:100], y.iloc[:100])
        scaler = model.estimator_.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, X.iloc[:100].mean().to_numpy())
        before = scaler.mean_.copy()
        model.predict(X.iloc[100:])
        np.testing.assert_array_equal(before, scaler.mean_)

    def test_candidates_clone_and_round_trip(self):
        X = example_features()
        y = np.maximum(0, 2 * X["lag_1"] - 10)
        specifications = [
            ("linear", {"feature_mode": "original", "basis": "quadratic"}),
            ("linear", {"feature_mode": "time24", "basis": "spline", "knots": 3}),
            ("forest", {"feature_mode": "context", "n_estimators": 3, "max_depth": 3}),
        ]
        for kind, spec in specifications:
            with self.subTest(kind=kind, spec=spec):
                model = clone(make_estimator(kind, spec, jobs=1)).fit(X.iloc[:130], y.iloc[:130])
                expected = model.predict(X.iloc[130:])
                self.assertTrue(np.isfinite(expected).all())
                self.assertTrue((expected >= 0).all())
                buffer = BytesIO()
                joblib.dump(model, buffer)
                buffer.seek(0)
                np.testing.assert_allclose(joblib.load(buffer).predict(X.iloc[130:]), expected)

    def test_clipping_is_saved_and_schema_is_validated(self):
        X = example_features()
        model = make_estimator("linear", {"basis": "linear"}).fit(X, 2 * X["lag_1"] + 3)
        new = X.iloc[:1].copy()
        new["lag_1"] = -100
        self.assertLess(model.predict_raw(new)[0], 0)
        self.assertEqual(model.predict(new)[0], 0)
        with self.assertRaisesRegex(ValueError, "15 named"):
            model.predict(X.iloc[:, ::-1])
        new["lag_1"] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            model.predict(new)


if __name__ == "__main__":
    unittest.main()
