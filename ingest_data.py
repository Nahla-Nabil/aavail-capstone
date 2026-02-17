"""
ingest_data.py
--------------
Data ingestion script for the AAVAIL Capstone Project.
Loads, cleans, and aggregates time-series revenue data.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def fetch_data(data_dir=DATA_DIR, countries=None):
    """
    Load all JSON files from data_dir, clean and return a combined DataFrame.
    
    Parameters
    ----------
    data_dir : str
        Directory containing the JSON data files.
    countries : list or None
        If provided, filter to only these countries.
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: country, date, revenue, purchases, unique_invoices, ...
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_data = []
    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    if not json_files:
        raise ValueError(f"No JSON files found in {data_dir}")

    for fname in json_files:
        fpath = os.path.join(data_dir, fname)
        try:
            with open(fpath, 'r') as f:
                data = json.load(f)
            
            # Handle both list and dict formats
            if isinstance(data, dict):
                data = [data]
            
            df = pd.DataFrame(data)
            
            # Normalize column names (some files may differ)
            df.columns = [c.lower().strip().replace(' ', '_') for c in df.columns]
            
            # Try to detect country from filename if not in data
            if 'country' not in df.columns:
                country_name = fname.replace('.json', '').replace('_', ' ').title()
                df['country'] = country_name
            
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {fname}: {e}")

    if not all_data:
        raise ValueError("No data could be loaded.")

    df_all = pd.concat(all_data, ignore_index=True)

    # Clean invoice ids (remove letters)
    if 'invoice' in df_all.columns:
        df_all['invoice'] = df_all['invoice'].astype(str).str.replace(r'[A-Za-z]', '', regex=True)

    # Parse dates
    date_col = None
    for col in ['invoice_date', 'date', 'invoicedate']:
        if col in df_all.columns:
            date_col = col
            break

    if date_col:
        df_all['date'] = pd.to_datetime(df_all[date_col], errors='coerce')
        df_all.dropna(subset=['date'], inplace=True)
    else:
        raise ValueError("No date column found in data.")

    # Filter by countries if specified
    if countries:
        df_all = df_all[df_all['country'].isin(countries)]

    # Compute revenue if not present
    if 'revenue' not in df_all.columns:
        if 'price' in df_all.columns and 'quantity' in df_all.columns:
            df_all['revenue'] = df_all['price'] * df_all['quantity']
        elif 'unit_price' in df_all.columns and 'quantity' in df_all.columns:
            df_all['revenue'] = df_all['unit_price'] * df_all['quantity']

    df_all = df_all[df_all['revenue'] > 0]

    return df_all


def engineer_features(df, country='all', training=True):
    """
    Aggregate daily revenue and engineer time-series features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw data from fetch_data().
    country : str
        Country to filter, or 'all' for combined.
    training : bool
        If True, return X and y for training.
    
    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray (only if training=True)
        Target revenue values.
    dates : list
        Dates corresponding to each row.
    """
    if country != 'all':
        df = df[df['country'].str.lower() == country.lower()]

    # Aggregate by date
    daily = df.groupby('date')['revenue'].sum().reset_index()
    daily = daily.sort_values('date').reset_index(drop=True)

    # Engineer lag features
    daily['revenue_lag1'] = daily['revenue'].shift(1)
    daily['revenue_lag7'] = daily['revenue'].shift(7)
    daily['revenue_lag30'] = daily['revenue'].shift(30)
    daily['revenue_rolling7'] = daily['revenue'].rolling(7).mean()
    daily['revenue_rolling30'] = daily['revenue'].rolling(30).mean()
    daily['month'] = daily['date'].dt.month
    daily['day_of_week'] = daily['date'].dt.dayofweek
    daily['year'] = daily['date'].dt.year

    daily.dropna(inplace=True)

    feature_cols = ['revenue_lag1', 'revenue_lag7', 'revenue_lag30',
                    'revenue_rolling7', 'revenue_rolling30',
                    'month', 'day_of_week', 'year']

    X = daily[feature_cols].values
    y = daily['revenue'].values
    dates = daily['date'].tolist()

    if training:
        return X, y, dates
    return X, dates


if __name__ == '__main__':
    print("Loading data...")
    df = fetch_data()
    print(f"Loaded {len(df)} rows")
    print(f"Countries: {df['country'].unique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(df.head())
