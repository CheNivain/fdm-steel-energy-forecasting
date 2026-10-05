"""Check new forecast transformations and weighted/residual model semantics."""

from io import BytesIO
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.advanced_training import ScheduleFeatures, make_model, missing_reading_sensitivity
from test_optimized_regression import example_features


class AdvancedChecks(unittest.TestCase):
    def setUp(self):
        self.X = example_features()
        self.X.index = pd.date_range("2018-01-01", periods=len(self.X), freq="15min")
        self.y = 2 * self.X.lag_1 + 4

    def test_schedule_features_are_row_local_and_load_removable(self):
        for schedule in ["none", "weekend", "harmonics"]:
            features = ScheduleFeatures("time96", False, schedule).fit(self.X)
            changed = self.X.copy()
            changed.load_level = 99
            pd.testing.assert_frame_equal(features.transform(self.X), features.transform(changed))
            pd.testing.assert_frame_equal(features.transform(self.X).iloc[[32]], features.transform(self.X.iloc[[32]]))

    def test_residual_prediction_adds_last_usage_and_survives_reload(self):
        model = make_model("forest", {"n_estimators": 4, "max_depth": 3, "target_mode": "delta", "high_weight": 1.3, "half_life_days": 90}, jobs=1)
        fitted = clone(model).fit(self.X.iloc[:120], self.y.iloc[:120])
        X = self.X.iloc[120:]
        expected = np.maximum(0, fitted.estimator_.predict(X) + X.lag_1.to_numpy())
        np.testing.assert_allclose(fitted.predict(X), expected)
        stream = BytesIO()
        joblib.dump(fitted, stream)
        stream.seek(0)
        np.testing.assert_allclose(joblib.load(stream).predict(X), expected)

    def test_linear_pipeline_scales_only_fitting_rows(self):
        model = make_model("linear", {"knots": 3, "high_weight": 2.0, "schedule": "none"})
        model.fit(self.X.iloc[:120], self.y.iloc[:120])
        scaler = model.estimator_.named_steps["scaler"]
        fitted_basis = model.estimator_[:-2].transform(self.X.iloc[:120])
        np.testing.assert_allclose(scaler.mean_, fitted_basis.mean(axis=0))
        before = scaler.mean_.copy()
        model.predict(self.X.iloc[120:])
        np.testing.assert_array_equal(before, scaler.mean_)

    def test_rejects_misaligned_training_targets(self):
        model = make_model("forest", {"n_estimators": 2}, jobs=1)
        y = self.y.copy()
        y.index += pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "do not align"):
            model.fit(self.X, y)

    def test_interpolated_target_and_all_known_dependencies_are_excluded(self):
        index = pd.date_range("2018-11-08", periods=674, freq="15min")
        frame = pd.DataFrame({"actual_kWh": 5.0, "predicted_kWh": 5.0}, index=index)
        frame.iloc[list(range(97)) + [672], 1] = 100
        result = missing_reading_sensitivity(frame)
        self.assertEqual(result["n_rows"], 576)
        self.assertEqual(result["RMSE_kWh"], 0)


if __name__ == "__main__":
    unittest.main()
