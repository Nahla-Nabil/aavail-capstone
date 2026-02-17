"""
app.py
------
Flask API for the AAVAIL Revenue Forecasting Capstone.
Endpoints: /train, /predict, /logs, /health
"""

import os
import sys
import json
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime

# Make sure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.ingest_data import fetch_data, engineer_features
from src.model import train_model, predict
from src.logger import log_prediction, log_training, load_log, monitor_performance, PREDICT_LOG, TRAIN_LOG

app = Flask(__name__)
app.config['TESTING'] = False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': str(datetime.now())})


@app.route('/train', methods=['POST'])
def train():
    """
    Train model for a given country.
    
    Body (JSON):
        country: str  (default='all')
    """
    body = request.get_json(force=True, silent=True) or {}
    country = body.get('country', 'all')

    try:
        df = fetch_data()
        X, y, dates = engineer_features(df, country=country, training=True)

        if len(X) < 10:
            return jsonify({'error': f'Not enough data for country: {country}'}), 400

        pipeline, results = train_model(X, y, country=country)

        # Log training
        best = min(results, key=lambda k: results[k]['mae'])
        log_training(
            country=country,
            mae=results[best]['mae'],
            rmse=results[best]['rmse'],
            n_samples=len(X),
            model_name=best
        )

        return jsonify({
            'status': 'success',
            'country': country,
            'samples_trained': int(len(X)),
            'model_comparison': results,
            'best_model': best
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['GET', 'POST'])
def make_prediction():
    """
    Predict next 30-day revenue for a country.
    
    GET params or POST JSON:
        country: str  (default='all')
        date: str     (YYYY-MM-DD, default=today)
    """
    if request.method == 'POST':
        body = request.get_json(force=True, silent=True) or {}
        country = body.get('country', 'all')
        date_str = body.get('date', str(datetime.now().date()))
    else:
        country = request.args.get('country', 'all')
        date_str = request.args.get('date', str(datetime.now().date()))

    try:
        df = fetch_data()
        X, dates = engineer_features(df, country=country, training=False)

        if len(X) == 0:
            return jsonify({'error': f'No data for country: {country}'}), 400

        # Use last row for prediction (most recent data point)
        X_input = X[-1:, :]
        prediction = predict(X_input, country=country)
        pred_value = float(prediction[0])

        # Scale to 30-day estimate
        pred_30day = pred_value * 30

        # Log prediction
        log_prediction(
            country=country,
            date=date_str,
            prediction=pred_30day
        )

        return jsonify({
            'country': country,
            'date': date_str,
            'predicted_revenue_30day': round(pred_30day, 2),
            'currency': 'GBP'
        })

    except FileNotFoundError:
        return jsonify({'error': f'No trained model found for country: {country}. Train first.'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Return prediction or training logs.
    
    GET params:
        type: 'predict' or 'train' (default='predict')
        n: int - number of recent records (default=10)
    """
    log_type = request.args.get('type', 'predict')
    n = int(request.args.get('n', 10))

    log_file = PREDICT_LOG if log_type == 'predict' else TRAIN_LOG
    logs = load_log(log_file)
    recent = logs[-n:]

    return jsonify({
        'type': log_type,
        'count': len(logs),
        'records': recent
    })


@app.route('/monitor', methods=['GET'])
def monitor():
    """
    Return performance monitoring metrics from log.
    """
    logs = load_log(PREDICT_LOG)
    perf_logs = [l for l in logs if l.get('type') == 'performance']

    return jsonify({
        'total_predictions': len([l for l in logs if l.get('type') == 'predict']),
        'performance_checks': perf_logs[-5:]
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
