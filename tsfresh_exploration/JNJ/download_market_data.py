import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pytz

def fetch_market_data():
    """Fetch and cache market data for strategy."""
    # Initialize variables
    symbols = ['JNJ', 'XLV']
    cache_file = 'data/market_data.pkl'
    os.makedirs('data', exist_ok=True)
    
    # Load API credentials
    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not api_secret:
        raise ValueError("API credentials not found in environment variables")
    
    # Initialize API
    api = tradeapi.REST(
        api_key,
        api_secret,
        base_url='https://paper-api.alpaca.markets',
        api_version='v2'
    )
    
    # Set time period (5 years)
    end = datetime.now(pytz.timezone('US/Pacific'))
    start = end - timedelta(days=365*5)
    
    print(f"Fetching data from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    # Fetch data for each symbol
    data = {}
    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")
        try:
            bars = api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d'),
                adjustment='raw'
            ).df
            
            # Remove timezone info
            bars.index = bars.index.tz_localize(None)
            data[symbol] = bars
            
            print(f"Got {len(bars)} hours of data")
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            raise
    
    # Cache data
    print("\nCaching market data...")
    pd.to_pickle(data, cache_file)
    print(f"Data saved to {cache_file}")
    
    return data

if __name__ == "__main__":
    data = fetch_market_data()
    
    # Print summary
    print("\nData Summary:")
    print("-" * 50)
    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(f"Date Range: {df.index[0]} to {df.index[-1]}")
        print(f"Total Hours: {len(df)}")
        print(f"Price Range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
