import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
import requests

def download_hourly_data(symbol: str, api_key: str, api_secret: str, base_path='.'):
    """Download hourly data for a symbol."""
    print(f"\nDownloading data for {symbol}...")
    
    try:
        # Initialize client with data API endpoint
        client = StockHistoricalDataClient(
            api_key, 
            api_secret,
            raw_data=True
        )
        
        # Calculate dates (max 2 years)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=729)  # 729 to be safe
        
        # Create request with adjustment for splits/dividends
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date,
            adjustment=Adjustment.ALL  # Adjust for splits and dividends
        )
        
        # Get data
        print("Sending request to Alpaca...")
        bars = client.get_stock_bars(request)
        print("Received response from Alpaca")
        
        # Convert to DataFrame
        df = pd.DataFrame(bars[symbol])  # Raw data mode returns dict
        
        # Print DataFrame info
        print("\nDataFrame Info:")
        print(df.info())
        print("\nColumns:", df.columns.tolist())
        print("\nSample data:")
        print(df.head())
        
        if len(df) == 0:
            raise ValueError(f"No data downloaded for {symbol}")
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(base_path, 'data/cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Rename columns to match strategy expectations
        column_map = {
            'c': 'close',
            'h': 'high',
            'l': 'low',
            'o': 'open',
            'v': 'volume',
            't': 'timestamp'
        }
        df = df.rename(columns=column_map)
        
        # Convert ISO timestamps to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Drop unused columns
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        # Save to cache
        cache_file = os.path.join(cache_dir, f'{symbol.lower()}_data.pkl')
        df.to_pickle(cache_file)
        
        print(f"\nProcessed DataFrame:")
        print(df.head())
        print(f"\nDownloaded {len(df)} bars")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Saved to {cache_file}")
        
        # Print some basic stats
        returns = df['close'].pct_change()
        vol = returns.rolling(window=20).std() * np.sqrt(252)
        print("\nVolatility Statistics:")
        print("-" * 30)
        print(f"Mean: {vol.mean():.1%}")
        print(f"Median: {vol.median():.1%}")
        for p in [25, 50, 67, 75, 90]:
            print(f"{p}th percentile: {vol.quantile(p/100):.1%}")
        
        return df
    
    except Exception as e:
        print(f"Error details:")
        print(f"Type: {type(e)}")
        print(f"Message: {str(e)}")
        if hasattr(e, 'response'):
            print(f"Status code: {e.response.status_code}")
            print(f"Response text: {e.response.text}")
        raise

def ensure_data_downloaded(symbols, api_key, api_secret, base_path='.', force_download=False):
    """Ensure we have data for all symbols, downloading if needed."""
    for symbol in symbols:
        cache_file = os.path.join(base_path, 'data/cache', f'{symbol.lower()}_data.pkl')
        if force_download or not os.path.exists(cache_file):
            try:
                print(f"Downloading data for {symbol}...")
                download_hourly_data(symbol, api_key, api_secret, base_path)
            except Exception as e:
                print(f"Error downloading {symbol}: {e}")
                continue
        else:
            print(f"Using cached data for {symbol}")

if __name__ == "__main__":
    # Load config
    with open('strategy/config.json', 'r') as f:
        config = json.load(f)
    
    print(f"Using API key: {config['alpaca']['api_key'][:5]}...")
    
    # Download data for all symbols in config
    ensure_data_downloaded(
        config['symbols'],
        config['alpaca']['api_key'],
        config['alpaca']['api_secret']
    )
