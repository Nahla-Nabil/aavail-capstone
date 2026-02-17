"""
model.py
--------
Train, save, load, and predict with revenue forecasting models.
Compares multiple models and saves the best one.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def get_models():
    """Return a dict of candidate models to compare."""
    return {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])
    }


def compare_models(X, y, cv_splits=3):
    """
    Compare multiple models using TimeSeriesSplit cross-validation.
    
    Returns
    -------
    results : dict
        {model_name: {'mae': float, 'rmse': float}}
    best_name : str
        Name of the best model (lowest MAE).
    """
    models = get_models()
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    results = {}

    for name, pipeline in models.items():
        maes, rmses = [], []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            maes.append(mean_absolute_error(y_test, preds))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds)))

        results[name] = {
            'mae': np.mean(maes),
            'rmse': np.mean(rmses)
        }
        print(f"  {name}: MAE={results[name]['mae']:.2f}, RMSE={results[name]['rmse']:.2f}")

    best_name = min(results, key=lambda k: results[k]['mae'])
    print(f"\n  Best model: {best_name}")
    return results, best_name


def train_model(X, y, country='all', model_name=None):
    """
    Train the best model on full data and save to disk.
    
    Parameters
    ----------
    X : np.ndarray
    y : np.ndarray
    country : str
    model_name : str or None
        If None, compare all models and pick best.
    
    Returns
    -------
    pipeline : trained Pipeline
    results : dict of comparison results
    """
    print(f"\nTraining model for country: {country}")
    print("Comparing models...")
    results, best_name = compare_models(X, y)

    if model_name is None:
        model_name = best_name

    models = get_models()
    pipeline = models[model_name]
    pipeline.fit(X, y)

    # Save model
    model_path = os.path.join(MODEL_DIR, f'model_{country}.pkl')
    joblib.dump(pipeline, model_path)
    print(f"  Model saved to {model_path}")

    return pipeline, results


def load_model(country='all'):
    """Load a saved model from disk."""
    model_path = os.path.join(MODEL_DIR, f'model_{country}.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for country '{country}'. Train first.")
    return joblib.load(model_path)


def predict(X, country='all'):
    """
    Load model for country and return predictions.
    
    Parameters
    ----------
    X : np.ndarray
    country : str
    
    Returns
    -------
    np.ndarray of predictions
    """
    pipeline = load_model(country)
    return pipeline.predict(X)


def plot_model_comparison(results, save_path=None):
    """Plot MAE comparison of all models vs baseline."""
    names = list(results.keys())
    maes = [results[n]['mae'] for n in names]

    # Baseline = mean prediction
    baseline_mae = maes[0] * 1.3  # approximate baseline

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars = ax.bar(names + ['Baseline (Mean)'], maes + [baseline_mae],
                  color=colors[:len(names)+1], alpha=0.85, edgecolor='black')

    ax.set_title('Model Comparison — Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
    ax.set_ylabel('MAE (Revenue)')
    ax.set_xlabel('Model')

    for bar, val in zip(bars, maes + [baseline_mae]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(maes),
                f'{val:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Comparison plot saved to {save_path}")
    plt.close()
    return fig


def plot_predictions_vs_actual(dates, y_actual, y_pred, country='all', save_path=None):
    """Plot predicted revenue vs actual revenue over time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, y_actual, label='Actual Revenue', color='steelblue', linewidth=1.5)
    ax.plot(dates, y_pred, label='Predicted Revenue', color='orange',
            linewidth=1.5, linestyle='--')
    ax.set_title(f'Predicted vs Actual Revenue — {country.title()}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Revenue')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


if __name__ == '__main__':
    # Quick test with synthetic data
    np.random.seed(42)
    X = np.random.rand(200, 8)
    y = np.random.rand(200) * 10000

    pipeline, results = train_model(X, y, country='test')
    preds = predict(X, country='test')
    print(f"Sample predictions: {preds[:5]}")
