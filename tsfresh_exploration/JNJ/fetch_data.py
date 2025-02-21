import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pytz

def fetch_data(symbols=['JNJ', 'XLV'], days=365*5):
    """Fetch historical data for multiple symbols, using cached data if available."""
    cache_file = 'data/market_data.pkl'
    os.makedirs('data', exist_ok=True)
    
    data = {}
    
    # Try to load cached data first
    if os.path.exists(cache_file):
        print("Loading cached market data...")
        try:
            cached_data = pd.read_pickle(cache_file)
            # Only use cached data that we need
            for symbol in symbols:
                if symbol in cached_data:
                    data[symbol] = cached_data[symbol]
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    # Check which symbols we still need
    missing_symbols = [s for s in symbols if s not in data]
    if not missing_symbols:
        return data
    
    print(f"Fetching data for: {', '.join(missing_symbols)}")
    
    # Fetch missing data
    if missing_symbols:
        load_dotenv()
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        base_url = 'https://paper-api.alpaca.markets'
        
        if not api_key or not api_secret:
            raise ValueError("API credentials not found in environment variables")
            
        api = tradeapi.REST(
            api_key,
            api_secret,
            base_url=base_url,
            api_version='v2'
        )
        
        end = datetime.now()
        start = end - timedelta(days=days)
        
        for symbol in missing_symbols:
            print(f"\nFetching {days} days of data for {symbol}...")
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
        
        # Cache all data
        print("Caching market data...")
        pd.to_pickle(data, cache_file)
    
    return data

if __name__ == "__main__":
    # Fetch data
    data = fetch_data()
    
    # Print summary
    print("\nData Summary:")
    print("-" * 50)
    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(f"Date Range: {df.index[0]} to {df.index[-1]}")
        print(f"Total Hours: {len(df)}")
        print(f"Price Range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        
        # Print yearly stats
        print("\nYearly Data Points:")
        yearly_counts = df.groupby(df.index.year).size()
        for year, count in yearly_counts.items():
            print(f"{year}: {count} hours")
