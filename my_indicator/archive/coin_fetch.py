#!/usr/bin/env python3

import requests
import pandas as pd

# ------------------------------------------------------
# 1) Configure your CoinAPI API key and parameters here
# ------------------------------------------------------
API_KEY = "7186AC48-4486-4B1B-95B9-74D1FB4788B1"  # <-- Replace with your real CoinAPI key
BASE_URL = "https://rest.coinapi.io/v1/ohlcv"
SYMBOL_ID = "BITSTAMP_SPOT_BTC_USD"  # For example: "COINBASE_SPOT_BTC_USD", "KRAKEN_SPOT_ETH_USD", etc.
PERIOD_ID = "1MIN"   # or "5MIN", "1HRS", "1DAY", etc.
LIMIT = 1000         # how many bars to fetch (max depends on your plan)

# ------------------------------------------------------
# 2) Make the request
# ------------------------------------------------------
def fetch_coinapi_ohlcv(symbol_id: str, period_id: str, limit: int) -> pd.DataFrame:
    """
    Fetch OHLCV data from CoinAPI for a specific symbol, timeframe, and limit.
    Returns a pandas DataFrame with columns:
      [timestamp, open, high, low, close, volume]
    """
    endpoint = f"{BASE_URL}/{symbol_id}/latest"
    params = {
        "period_id": period_id,
        "limit": limit
    }
    
    headers = {
        "X-CoinAPI-Key": API_KEY
    }
    
    print(f"Fetching {limit} bars of {symbol_id} at period {period_id} from CoinAPI...")
    response = requests.get(endpoint, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.text}")
    
    data = response.json()
    if not data:
        print("No data returned. Check symbol_id or period_id.")
        return pd.DataFrame()
    
    # Convert the JSON response to a DataFrame
    df = pd.DataFrame(data)
    
    # The columns can vary, but typically you'll see:
    #   "time_period_start", "time_period_end", "time_open", "time_close",
    #   "price_open", "price_close", "price_high", "price_low", "volume_traded", etc.

    # Rename columns to a standard format
    df.rename(columns={
        "time_period_start": "timestamp",
        "price_open": "open",
        "price_high": "high",
        "price_low": "low",
        "price_close": "close",
        "volume_traded": "volume"
    }, inplace=True)
    
    # Convert timestamps to pandas datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Set timestamp as the index (optional)
    df.set_index("timestamp", inplace=True)
    
    # Sort by ascending time just in case
    df.sort_index(inplace=True)
    
    # Convert numeric columns to float (CoinAPI returns them as floats, but just to be sure)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)
    
    return df


def main():
    df = fetch_coinapi_ohlcv(SYMBOL_ID, PERIOD_ID, LIMIT)
    if df.empty:
        print("No data or empty DataFrame.")
        return

    # Print a sample of the data
    print("\nFetched Data (last 5 bars):")
    print(df.tail(5))
    
    # Example: Access typical columns
    latest_bar = df.iloc[-1]
    print(f"\nLatest Bar:\n{latest_bar.to_dict()}")
    
    # If you want to do further processing here, you can:
    #   - pass df into your strategy
    #   - compute indicators like MFI, OBV, etc.
    #   - store df in a CSV or database for caching

if __name__ == "__main__":
    main()
