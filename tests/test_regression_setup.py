"""Verify data alignment and check-only execution without fitting any estimator."""

from contextlib import ExitStack, chdir, redirect_stdout
from io import StringIO
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import regression_common as common


class InputChecks(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2018-10-31 23:30", periods=4, freq="15min", name="timestamp")
        X = pd.DataFrame(1.0, index=index, columns=common.FEATURES)
        y = pd.DataFrame({"Usage_kWh": [1.0] * 4}, index=index)
        self.tables = {
            "X_train": X.iloc[:2].copy(), "X_test": X.iloc[2:].copy(),
            "y_train": y.iloc[:2].copy(), "y_test": y.iloc[2:].copy(),
        }

    def load_fixture(self):
        with patch.object(Path, "is_file", return_value=True):
            with patch.object(common.pd, "read_csv", side_effect=lambda p, **kw: self.tables[p.stem].copy()):
                return common.load_data(Path("unused-fixture-path"))

    def test_valid_alignment(self):
        X_train, X_test, y_train, y_test = self.load_fixture()
        self.assertTrue(X_train.index.equals(y_train.index))
        self.assertTrue(X_test.index.equals(y_test.index))

    def test_rejects_misaligned_answers(self):
        self.tables["y_test"].index += pd.Timedelta(minutes=15)
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.load_fixture()

    def test_rejects_missing_values(self):
        self.tables["X_train"].iloc[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "missing or infinite"):
            self.load_fixture()

    def test_rejects_current_target_as_previous_reading(self):
        self.tables["X_test"].loc[:, "lag_1"] = 2
        with self.assertRaisesRegex(ValueError, "previous actual usage"):
            self.load_fixture()

    def test_rejects_shuffled_feature_columns(self):
        self.tables["X_test"] = self.tables["X_test"].iloc[:, ::-1]
        with self.assertRaisesRegex(ValueError, "saved order"):
            self.load_fixture()

    def test_rejects_reversed_time_order(self):
        self.tables["X_train"] = self.tables["X_train"].iloc[::-1]
        with self.assertRaisesRegex(ValueError, "chronological"):
            self.load_fixture()

    def test_validation_boundary(self):
        index = pd.date_range("2018-08-31 23:45", periods=3, freq="15min")
        X = pd.DataFrame({"lag_1": [1, 2, 3]}, index=index)
        y = pd.Series([2, 3, 4], index=index)
        X_fit, X_val, y_fit, y_val = common.split_for_validation(X, y)
        self.assertEqual(len(X_fit), 1)
        self.assertEqual(len(X_val), 2)
        self.assertLess(X_fit.index.max(), X_val.index.min())
        self.assertTrue(X_fit.index.equals(y_fit.index))
        self.assertTrue(X_val.index.equals(y_val.index))

    def test_metrics_against_hand_calculation(self):
        result = common.score_predictions("Example", "validation", [1, 2, 3], [1, 2, 4])
        self.assertAlmostEqual(result["MAE_kWh"], 1 / 3)
        self.assertAlmostEqual(result["RMSE_kWh"], np.sqrt(1 / 3))
        self.assertAlmostEqual(result["R2"], 0.5)


class CheckOnlyExecution(unittest.TestCase):
    def block_training(self, stack):
        for estimator in [LinearRegression, RandomForestRegressor, StandardScaler, Pipeline]:
            stack.enter_context(patch.object(estimator, "fit", side_effect=AssertionError("Check mode attempted training")))
        stack.enter_context(patch.object(common, "save_results", side_effect=AssertionError("Check mode attempted output export")))
        stack.enter_context(redirect_stdout(StringIO()))

    def test_scripts_do_not_fit_or_export(self):
        for folder in ["linear regression", "random forest"]:
            with self.subTest(folder=folder), ExitStack() as stack:
                self.block_training(stack)
                path = ROOT / folder / "model.py"
                stack.enter_context(patch.object(sys, "argv", [str(path), "--check"]))
                namespace = runpy.run_path(str(path), run_name="__main__")
                self.assertFalse(namespace["RUN_TRAINING"])
                self.assertNotIn("final_model", namespace)
                with self.assertRaises(NotFittedError):
                    check_is_fitted(namespace["model"])

    def test_notebook_cells_run_without_training(self):
        for folder, filename in [("linear regression", "linear_regression.ipynb"), ("random forest", "random_forest.ipynb")]:
            with self.subTest(folder=folder), ExitStack() as stack:
                self.block_training(stack)
                stack.enter_context(chdir(ROOT / folder))
                notebook = nbformat.read(ROOT / folder / filename, as_version=4)
                nbformat.validate(notebook)
                namespace = {"__name__": "__main__"}
                for index, cell in enumerate(notebook.cells):
                    if cell.cell_type == "code":
                        self.assertIsNone(cell.execution_count)
                        self.assertEqual(cell.outputs, [])
                        exec(compile(cell.source, f"{filename}:cell{index}", "exec"), namespace)
                self.assertFalse(namespace["RUN_TRAINING"])
                self.assertNotIn("final_model", namespace)
                with self.assertRaises(NotFittedError):
                    check_is_fitted(namespace["model"])


if __name__ == "__main__":
    unittest.main()
