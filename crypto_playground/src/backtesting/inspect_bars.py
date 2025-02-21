import requests
import os
import json
from datetime import datetime, timedelta, UTC
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Load environment variables
load_dotenv()

# Get API credentials from environment
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')


def get_crypto_bars_5min(symbols, start, end):
    """
    Fetches 5-minute historical crypto bars from Alpaca's Crypto API.
    
    :param symbols: A list of symbols (e.g. ["BTCUSD", "ETHUSD"])
    :param start: Start time in ISO format (RFC-3339)
    :param end: End time in ISO format (RFC-3339)
    :return: JSON response from the Alpaca endpoint
    """
    url = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
    
    # Fetch data for each symbol separately and combine results
    combined_data = {"bars": {}}
    
    for symbol in symbols:
        params = {
            "symbols": symbol,  # Single symbol at a time
            "timeframe": "5Min",
            "start": start,
            "end": end,
            "limit": 10000
        }

        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
        }

        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"Error fetching {symbol}: {response.status_code}, {response.text}")
            continue
            
        data = response.json()
        if data.get('bars'):
            combined_data['bars'].update(data['bars'])
    
    return combined_data

def process_and_save_bars(data, output_dir='data'):
    """
    Process the raw bar data and save it to CSV files.
    
    :param data: JSON response from Alpaca API
    :param output_dir: Directory to save the processed data
    """
    # Ensure output directory exists
    os.makedirs(f'src/backtesting/{output_dir}', exist_ok=True)
    
    # Process each symbol's data
    for symbol, bars in data.get('bars', {}).items():
        if not bars:
            print(f"Warning: No data received for {symbol}")
            continue
            
        print(f"\nProcessing {len(bars)} bars for {symbol}...")
            
        # Convert to DataFrame
        df = pd.DataFrame(bars)
        
        # Rename 't' to 'timestamp' and convert to datetime
        df = df.rename(columns={'t': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Rename price columns to match expected format
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'trades'
        })
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Save to CSV
        clean_symbol = symbol.replace('/', '')
        filename = f'src/backtesting/{output_dir}/{clean_symbol}_5min.csv'
        df.to_csv(filename)
        print(f"Saved {len(df)} bars for {symbol} to {filename}")

def fetch_and_save_bars(lookback_days=30):
    """
    Fetch bars for the past N days and save them to CSV files.
    
    :param lookback_days: Number of days of historical data to fetch
    """
    # Calculate time range
    end_time = datetime.now(UTC)
    start_time = end_time - timedelta(days=lookback_days)
    
    # Format dates as YYYY-MM-DD
    start_iso = start_time.strftime("%Y-%m-%d")
    end_iso = end_time.strftime("%Y-%m-%d")
    
    try:
        # Use explicit list of symbols
        symbols = ["BTC/USD", "ETH/USD"]
        print("Using symbols:", symbols)
        
        # Fetch historical bars
        print(f"Fetching {lookback_days} days of 5-minute bars...")
        crypto_data = get_crypto_bars_5min(symbols, start_iso, end_iso)
    except json.JSONDecodeError as e:
        print(f"Error parsing symbols from environment: {symbols_str}")
        print(f"Using default symbols: BTC/USD, ETH/USD")
        formatted_symbols = ["BTC/USD", "ETH/USD"]
        crypto_data = get_crypto_bars_5min(formatted_symbols, start_iso, end_iso)
    
    # Print raw response structure
    print("\nAPI Response Structure:")
    print(json.dumps(crypto_data, indent=2)[:1000])  # First 1000 chars for preview
    
    # Check if we got any data
    if not crypto_data.get('bars'):
        print("Error: No bars data in API response")
        print("Full response:", json.dumps(crypto_data, indent=2))
        return
        
    # Check which symbols we got data for
    received_symbols = set(crypto_data['bars'].keys())
    missing_symbols = set(symbols) - received_symbols
    if missing_symbols:
        print(f"\nWarning: No data received for symbols: {missing_symbols}")
    
    # Process and save the data
    process_and_save_bars(crypto_data)

if __name__ == "__main__":
    fetch_and_save_bars()
