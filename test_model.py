"""
tests/test_model.py
-------------------
Unit tests for model training and prediction.
Uses synthetic data — isolated from production models.
"""

import os
import sys
import unittest
import numpy as np
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import get_models, compare_models, train_model, load_model, predict


class TestModelComparison(unittest.TestCase):
    """Test that multiple models can be compared."""

    def test_get_models_returns_multiple(self):
        models = get_models()
        self.assertGreater(len(models), 1)

    def test_model_names_present(self):
        models = get_models()
        self.assertIn('RandomForest', models)
        self.assertIn('LinearRegression', models)
        self.assertIn('GradientBoosting', models)

    def test_compare_models_returns_results(self):
        X = np.random.rand(100, 8)
        y = np.random.rand(100) * 10000
        results, best_name = compare_models(X, y, cv_splits=2)
        self.assertIsInstance(results, dict)
        self.assertIn(best_name, results)

    def test_compare_models_has_metrics(self):
        X = np.random.rand(100, 8)
        y = np.random.rand(100) * 10000
        results, _ = compare_models(X, y, cv_splits=2)
        for name, metrics in results.items():
            self.assertIn('mae', metrics)
            self.assertIn('rmse', metrics)

    def test_mae_is_positive(self):
        X = np.random.rand(100, 8)
        y = np.random.rand(100) * 10000
        results, _ = compare_models(X, y, cv_splits=2)
        for name, metrics in results.items():
            self.assertGreater(metrics['mae'], 0)


class TestModelTrainAndLoad(unittest.TestCase):
    """Test model training, saving, and loading with isolated temp directory."""

    def setUp(self):
        # Use a temp directory to avoid touching production models
        self.tmp_dir = tempfile.mkdtemp()
        # Patch MODEL_DIR for tests
        import src.model as m
        self._orig_model_dir = m.MODEL_DIR
        m.MODEL_DIR = self.tmp_dir

    def tearDown(self):
        import src.model as m
        m.MODEL_DIR = self._orig_model_dir
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_train_model_saves_file(self):
        import src.model as m
        X = np.random.rand(100, 8)
        y = np.random.rand(100) * 10000
        m.train_model(X, y, country='test_country')
        self.assertTrue(os.path.exists(os.path.join(self.tmp_dir, 'model_test_country.pkl')))

    def test_load_model_after_training(self):
        import src.model as m
        X = np.random.rand(100, 8)
        y = np.random.rand(100) * 10000
        m.train_model(X, y, country='test_country')
        pipeline = m.load_model(country='test_country')
        self.assertIsNotNone(pipeline)

    def test_load_model_raises_if_not_trained(self):
        import src.model as m
        with self.assertRaises(FileNotFoundError):
            m.load_model(country='nonexistent_country_xyz')

    def test_predict_returns_array(self):
        import src.model as m
        X = np.random.rand(100, 8)
        y = np.random.rand(100) * 10000
        m.train_model(X, y, country='test_country')
        X_test = np.random.rand(5, 8)
        preds = m.predict(X_test, country='test_country')
        self.assertEqual(len(preds), 5)

    def test_predict_values_are_numeric(self):
        import src.model as m
        X = np.random.rand(100, 8)
        y = np.abs(np.random.rand(100) * 10000)
        m.train_model(X, y, country='test_country')
        X_test = np.random.rand(3, 8)
        preds = m.predict(X_test, country='test_country')
        for p in preds:
            self.assertIsInstance(float(p), float)

    def test_train_returns_results_dict(self):
        import src.model as m
        X = np.random.rand(100, 8)
        y = np.random.rand(100) * 10000
        pipeline, results = m.train_model(X, y, country='test_country')
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)


if __name__ == '__main__':
    unittest.main()
