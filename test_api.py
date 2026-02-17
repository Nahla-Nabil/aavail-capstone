"""
tests/test_api.py
-----------------
Unit tests for the Flask API endpoints.
Uses mock data — isolated from production models and logs.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import app


class TestAPIHealth(unittest.TestCase):
    """Test the /health endpoint."""

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_health_status_200(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)

    def test_health_returns_ok(self):
        response = self.client.get('/health')
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')

    def test_health_has_timestamp(self):
        response = self.client.get('/health')
        data = json.loads(response.data)
        self.assertIn('timestamp', data)


class TestAPIPredict(unittest.TestCase):
    """Test the /predict endpoint."""

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    @patch('app.fetch_data')
    @patch('app.engineer_features')
    @patch('app.predict')
    @patch('app.log_prediction')
    def test_predict_all_countries(self, mock_log, mock_predict, mock_features, mock_fetch):
        """Test prediction for all countries combined."""
        mock_fetch.return_value = MagicMock()
        mock_features.return_value = (np.random.rand(10, 8), ['2019-01-01'] * 10)
        mock_predict.return_value = np.array([500.0])

        response = self.client.get('/predict?country=all')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('predicted_revenue_30day', data)
        self.assertEqual(data['country'], 'all')

    @patch('app.fetch_data')
    @patch('app.engineer_features')
    @patch('app.predict')
    @patch('app.log_prediction')
    def test_predict_specific_country(self, mock_log, mock_predict, mock_features, mock_fetch):
        """Test prediction for a specific country."""
        mock_fetch.return_value = MagicMock()
        mock_features.return_value = (np.random.rand(10, 8), ['2019-01-01'] * 10)
        mock_predict.return_value = np.array([750.0])

        response = self.client.get('/predict?country=united_kingdom')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['country'], 'united_kingdom')

    @patch('app.fetch_data')
    @patch('app.engineer_features')
    @patch('app.predict')
    @patch('app.log_prediction')
    def test_predict_post_method(self, mock_log, mock_predict, mock_features, mock_fetch):
        """Test POST request to /predict."""
        mock_fetch.return_value = MagicMock()
        mock_features.return_value = (np.random.rand(10, 8), ['2019-01-01'] * 10)
        mock_predict.return_value = np.array([600.0])

        response = self.client.post(
            '/predict',
            data=json.dumps({'country': 'germany', 'date': '2019-03-15'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['country'], 'germany')

    @patch('app.fetch_data')
    @patch('app.engineer_features')
    @patch('app.predict')
    def test_predict_no_model_returns_404(self, mock_predict, mock_features, mock_fetch):
        """Test that missing model returns 404."""
        mock_fetch.return_value = MagicMock()
        mock_features.return_value = (np.random.rand(10, 8), ['2019-01-01'] * 10)
        mock_predict.side_effect = FileNotFoundError("No model found")

        response = self.client.get('/predict?country=nonexistent_country')
        self.assertEqual(response.status_code, 404)

    @patch('app.fetch_data')
    @patch('app.engineer_features')
    def test_predict_no_data_returns_400(self, mock_features, mock_fetch):
        """Test that empty data returns 400."""
        mock_fetch.return_value = MagicMock()
        mock_features.return_value = (np.array([]).reshape(0, 8), [])

        response = self.client.get('/predict?country=empty_country')
        self.assertEqual(response.status_code, 400)

    @patch('app.fetch_data')
    @patch('app.engineer_features')
    @patch('app.predict')
    @patch('app.log_prediction')
    def test_predict_returns_revenue_value(self, mock_log, mock_predict, mock_features, mock_fetch):
        """Test that prediction value is positive."""
        mock_fetch.return_value = MagicMock()
        mock_features.return_value = (np.random.rand(10, 8), ['2019-01-01'] * 10)
        mock_predict.return_value = np.array([1000.0])

        response = self.client.get('/predict')
        data = json.loads(response.data)
        self.assertGreater(data['predicted_revenue_30day'], 0)


class TestAPILogs(unittest.TestCase):
    """Test the /logs endpoint."""

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    @patch('app.load_log')
    def test_logs_predict_endpoint(self, mock_load):
        mock_load.return_value = [{'type': 'predict', 'country': 'all'}]
        response = self.client.get('/logs?type=predict')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('records', data)

    @patch('app.load_log')
    def test_logs_train_endpoint(self, mock_load):
        mock_load.return_value = [{'type': 'train', 'country': 'all'}]
        response = self.client.get('/logs?type=train')
        self.assertEqual(response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
