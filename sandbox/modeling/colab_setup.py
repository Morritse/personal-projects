# GPU-Optimization-Colab.ipynb

# First, check if GPU is available
!nvidia-smi

# Install required packages
!pip install cudf-cu11 cupy-cuda11x cuml-cu11 xgboost

# Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging

# Download historical data
def fetch_data(symbol, lookback_days=500):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    
    # Calculate basic features
    df['returns'] = df['Close'].pct_change()
    df['returns_volatility'] = df['returns'].rolling(20).std()
    
    # Add technical indicators (your existing calculation code)
    # ... 
    
    return df

# Run optimization
symbols = ['SPY', 'QQQ', 'AAPL']  # Add your symbols
results = {}

for symbol in symbols:
    print(f"\nOptimizing {symbol}")
    data = fetch_data(symbol)
    results[symbol] = run_gpu_optimization(data, symbol)
    
    print(f"\nResults for {symbol}:")
    print(f"RF Score: {results[symbol]['rf_score']:.4f}")
    print(f"XGB Score: {results[symbol]['xgb_score']:.4f}")

# Save results to Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

for symbol, result in results.items():
    filename = f"/content/drive/MyDrive/trading_optimization_{symbol}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(result, f)