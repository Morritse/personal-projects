import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi
import pytz

def download_historical_data():
    """Download 5 years of hourly data for JNJ and XLV."""
    # Initialize API
    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not api_secret:
        raise ValueError("API credentials not found in environment variables")
    
    api = tradeapi.REST(
        api_key,
        api_secret,
        base_url='https://paper-api.alpaca.markets',
        api_version='v2'
    )
    
    # Set time period (5 years)
    end = datetime.now(pytz.timezone('US/Pacific'))
    start = end - timedelta(days=365*5)
    
    # Core symbols only
    symbols = ['JNJ', 'XLV']
    
    # Create data directory if it doesn't exist
    os.makedirs('data/historical', exist_ok=True)
    
    # Download data for each symbol
    for symbol in symbols:
        print(f"\nDownloading {symbol}...")
        try:
            # Get hourly bars
            bars = api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d'),
                adjustment='raw'
            ).df
            
            # Save to CSV
            filename = f'data/historical/{symbol.lower()}_hourly.csv'
            bars.to_csv(filename)
            print(f"Saved {len(bars)} bars to {filename}")
            print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
            
            # Print some stats
            print("\nData Statistics:")
            print(f"Trading days: {len(bars.index.date.unique())}")
            print(f"Average bars per day: {len(bars)/len(bars.index.date.unique()):.1f}")
            print("\nHourly distribution:")
            print(bars.index.hour.value_counts().sort_index())
            
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            continue

if __name__ == "__main__":
    download_historical_data()
