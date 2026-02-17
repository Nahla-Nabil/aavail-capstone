"""
logger.py
---------
Logging and performance monitoring for the AAVAIL capstone.
Tracks predictions, timestamps, and model performance metrics.
"""

import os
import json
import numpy as np
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

PREDICT_LOG = os.path.join(LOG_DIR, 'predict_log.json')
TRAIN_LOG = os.path.join(LOG_DIR, 'train_log.json')


def update_log(data, log_file=PREDICT_LOG):
    """
    Append a record to a JSON log file.
    
    Parameters
    ----------
    data : dict
        Record to log.
    log_file : str
        Path to log file.
    """
    data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logs = load_log(log_file)
    logs.append(data)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2, default=str)


def load_log(log_file=PREDICT_LOG):
    """
    Load all records from a JSON log file.
    
    Parameters
    ----------
    log_file : str
    
    Returns
    -------
    list of dicts
    """
    if not os.path.exists(log_file):
        return []
    with open(log_file, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def log_prediction(country, date, prediction, model_version='1.0', log_file=PREDICT_LOG):
    """Log a single prediction event."""
    record = {
        'type': 'predict',
        'country': country,
        'date': str(date),
        'prediction': float(prediction) if hasattr(prediction, '__float__') else prediction,
        'model_version': model_version
    }
    update_log(record, log_file)


def log_training(country, mae, rmse, n_samples, model_name='RandomForest', log_file=TRAIN_LOG):
    """Log a model training event."""
    record = {
        'type': 'train',
        'country': country,
        'model_name': model_name,
        'mae': float(mae),
        'rmse': float(rmse),
        'n_samples': int(n_samples)
    }
    update_log(record, log_file)


def compute_wasserstein_distance(y1, y2):
    """
    Compute a simple approximation of the Wasserstein distance
    to detect distribution drift between two arrays.
    
    Parameters
    ----------
    y1, y2 : array-like
        Two distributions to compare.
    
    Returns
    -------
    float
    """
    y1 = np.sort(np.array(y1, dtype=float))
    y2 = np.sort(np.array(y2, dtype=float))

    # Interpolate to same length
    n = max(len(y1), len(y2))
    y1_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y1)), y1)
    y2_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y2)), y2)

    return float(np.mean(np.abs(y1_interp - y2_interp)))


def monitor_performance(y_actual, y_predicted, log_file=PREDICT_LOG):
    """
    Compute performance metrics and log them.
    
    Parameters
    ----------
    y_actual : array-like
    y_predicted : array-like
    log_file : str
    
    Returns
    -------
    dict with mae, rmse, drift
    """
    y_actual = np.array(y_actual, dtype=float)
    y_predicted = np.array(y_predicted, dtype=float)

    mae = float(np.mean(np.abs(y_actual - y_predicted)))
    rmse = float(np.sqrt(np.mean((y_actual - y_predicted) ** 2)))
    drift = compute_wasserstein_distance(y_actual, y_predicted)

    metrics = {
        'type': 'performance',
        'mae': mae,
        'rmse': rmse,
        'drift_score': drift,
        'n_samples': len(y_actual)
    }
    update_log(metrics, log_file)
    return metrics


if __name__ == '__main__':
    print("Testing logger...")
    log_prediction('united_kingdom', '2019-01-15', 12345.67)
    log_training('united_kingdom', mae=500.0, rmse=700.0, n_samples=365)
    
    y_actual = np.random.rand(100) * 10000
    y_pred = y_actual + np.random.rand(100) * 500
    metrics = monitor_performance(y_actual, y_pred)
    print(f"Performance metrics: {metrics}")
    print(f"Predict log entries: {len(load_log(PREDICT_LOG))}")
    print(f"Train log entries: {len(load_log(TRAIN_LOG))}")
