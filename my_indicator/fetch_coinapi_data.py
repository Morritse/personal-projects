import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# CoinAPI configuration
COINAPI_KEY = "7186AC48-4486-4B1B-95B9-74D1FB4788B1"
COINAPI_REST_URL = "https://rest.coinapi.io/v1/ohlcv"
PERIOD_ID = "1MIN"
SYMBOL_ID = "COINBASE_SPOT_ETH_USD"  # Using Coinbase as reference exchange

def fetch_historical_data(symbol_id: str, period_id: str, limit: int) -> pd.DataFrame:
    """Fetch up to 'limit' 1-min bars for 'symbol_id' from CoinAPI via REST."""
    endpoint = f"{COINAPI_REST_URL}/{symbol_id}/history"
    headers = {"X-CoinAPI-Key": COINAPI_KEY}
    params = {
        "period_id": period_id,
        "limit": limit
    }
    print(f"[REST] Fetching {limit} {period_id} bars for {symbol_id} ...")
    
    response = requests.get(endpoint, headers=headers, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        print(response.text)
        return None

def process_ohlcv_data(data):
    """Convert API response to pandas DataFrame"""
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    if df.empty:
        return df
        
    # Rename columns to standard
    df.rename(columns={
        "time_period_start": "time_period_start",
        "time_period_end":   "time_period_end",
        "time_open":         "time_open",
        "time_close":        "time_close",
        "price_open":        "price_open",
        "price_high":        "price_high",
        "price_low":         "price_low",
        "price_close":       "price_close",
        "volume_traded":     "volume_traded",
    }, inplace=True)
    
    # Convert to datetime
    df["time_period_start"] = pd.to_datetime(df["time_period_start"], utc=True)
    df["time_period_end"]   = pd.to_datetime(df["time_period_end"],   utc=True)
    df["time_open"]         = pd.to_datetime(df["time_open"],         utc=True)
    df["time_close"]        = pd.to_datetime(df["time_close"],        utc=True)
    
    # Sort ascending by start time
    df.sort_values("time_period_start", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def main():
    # Create cache directory if it doesn't exist
    os.makedirs('cache_data', exist_ok=True)
    
    # Fetch historical data
    raw_data = fetch_historical_data(SYMBOL_ID, PERIOD_ID, limit=50000)
    
    # Process data
    df = process_ohlcv_data(raw_data)
    
    if not df.empty:
        # Convert to backtrader format
        bt_df = df.copy()
        bt_df.rename(columns={
            "price_open": "open",
            "price_high": "high",
            "price_low": "low",
            "price_close": "close",
            "volume_traded": "volume"
        }, inplace=True)
        bt_df.set_index("time_period_start", inplace=True)
        bt_df = bt_df[['open', 'high', 'low', 'close', 'volume']]
        
        # Save to file
        output_file = 'cache_data/eth_usd_coinapi.csv'
        bt_df.to_csv(output_file)
        print(f"Data saved to {output_file}")
        print(f"Total records: {len(bt_df)}")
        print("\nSample data:")
        print(bt_df.head())
    else:
        print("No data to save")

if __name__ == '__main__':
    main()
