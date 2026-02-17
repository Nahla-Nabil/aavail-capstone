"""
tests/test_logger.py
---------------------
Unit tests for logging and performance monitoring.
Uses temporary log files — isolated from production logs.
"""

import os
import sys
import json
import unittest
import tempfile
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.logger import (
    update_log, load_log,
    log_prediction, log_training,
    compute_wasserstein_distance,
    monitor_performance
)


class TestLogReadWrite(unittest.TestCase):
    """Test log file creation, write, and read — isolated from production."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.test_log = os.path.join(self.tmp_dir, 'test_log.json')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_log_creates_file(self):
        update_log({'value': 42}, log_file=self.test_log)
        self.assertTrue(os.path.exists(self.test_log))

    def test_load_empty_log(self):
        result = load_log(self.test_log)
        self.assertEqual(result, [])

    def test_update_and_load_log(self):
        update_log({'revenue': 1000}, log_file=self.test_log)
        logs = load_log(self.test_log)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['revenue'], 1000)

    def test_multiple_entries(self):
        update_log({'revenue': 1000}, log_file=self.test_log)
        update_log({'revenue': 2000}, log_file=self.test_log)
        update_log({'revenue': 3000}, log_file=self.test_log)
        logs = load_log(self.test_log)
        self.assertEqual(len(logs), 3)

    def test_log_has_timestamp(self):
        update_log({'value': 1}, log_file=self.test_log)
        logs = load_log(self.test_log)
        self.assertIn('timestamp', logs[0])

    def test_log_prediction_entry(self):
        log_prediction('united_kingdom', '2019-01-15', 12345.67, log_file=self.test_log)
        logs = load_log(self.test_log)
        self.assertEqual(logs[0]['country'], 'united_kingdom')
        self.assertEqual(logs[0]['prediction'], 12345.67)

    def test_log_training_entry(self):
        log_training('germany', mae=500.0, rmse=700.0, n_samples=300, log_file=self.test_log)
        logs = load_log(self.test_log)
        self.assertEqual(logs[0]['country'], 'germany')
        self.assertEqual(logs[0]['mae'], 500.0)


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring and drift detection."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.test_log = os.path.join(self.tmp_dir, 'perf_log.json')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_wasserstein_same_distributions(self):
        """Same distribution should have near-zero distance."""
        y = np.random.rand(100) * 1000
        dist = compute_wasserstein_distance(y, y)
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_wasserstein_different_distributions(self):
        """Different distributions should have positive distance."""
        y1 = np.zeros(100)
        y2 = np.ones(100) * 1000
        dist = compute_wasserstein_distance(y1, y2)
        self.assertGreater(dist, 0)

    def test_monitor_returns_metrics(self):
        y_actual = np.random.rand(50) * 10000
        y_pred = y_actual + np.random.rand(50) * 500
        metrics = monitor_performance(y_actual, y_pred, log_file=self.test_log)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('drift_score', metrics)

    def test_monitor_logs_to_file(self):
        y_actual = np.random.rand(50) * 10000
        y_pred = y_actual + np.random.rand(50) * 500
        monitor_performance(y_actual, y_pred, log_file=self.test_log)
        logs = load_log(self.test_log)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]['type'], 'performance')

    def test_mae_is_non_negative(self):
        y_actual = np.random.rand(50) * 10000
        y_pred = y_actual + np.random.rand(50) * 500
        metrics = monitor_performance(y_actual, y_pred, log_file=self.test_log)
        self.assertGreaterEqual(metrics['mae'], 0)

    def test_perfect_predictions_mae_zero(self):
        y = np.array([100.0, 200.0, 300.0])
        metrics = monitor_performance(y, y, log_file=self.test_log)
        self.assertAlmostEqual(metrics['mae'], 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
