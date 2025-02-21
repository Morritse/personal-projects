#!/usr/bin/env python3

import requests
import datetime
import pandas as pd
import os

# ------------------------------------------
# Configuration
# ------------------------------------------
API_KEY = "PKASZ48REAQARDXG66WF"
API_SECRET = "L8w2jmhDilnFSxFA9VLNMDbef0copxhf3NOTXSFH"

# Alpaca's Market Data v2 base endpoint for stocks/ETFs:
BASE_URL = "https://data.alpaca.markets/v2"

# The 15 ETFs you'd like to fetch:
ETF_SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "TLT", "IEF", "GLD",
    "SLV", "USO", "UNG", "DBC",
    "UUP", "LQD", "XLE"
]

# How many years of data to retrieve:
YEARS_TO_FETCH = 2

# ------------------------------------------
# Create output folder
# ------------------------------------------
OUTPUT_DIR = "working_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------
# Date range
# ------------------------------------------
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=YEARS_TO_FETCH * 365)
# Format as ISO 8601 strings (UTC)
start_str = start_date.strftime("%Y-%m-%dT%H:%M:%SZ")
end_str = end_date.strftime("%Y-%m-%dT%H:%M:%SZ")

# ------------------------------------------
# Common request headers for authentication
# ------------------------------------------
HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

# ------------------------------------------
# Fetch function
# ------------------------------------------
def fetch_data(symbol: str):
    """
    Fetch daily bars for a given symbol from Alpaca's Data API.
    Returns a pandas DataFrame with columns: [Date, Open, High, Low, Close, Volume].
    """
    url = f"{BASE_URL}/stocks/{symbol}/bars"
    
    params = {
        "timeframe": "1Day",      # daily bars
        "start": start_str,
        "end": end_str,
        "adjustment": "all",      # get split and dividend adjusted data
        "limit": 10000           # in case of many bars
    }
    
    resp = requests.get(url, headers=HEADERS, params=params)
    resp.raise_for_status()  # will raise an exception if there's an error
    data = resp.json()
    
    # 'data' should be a dict with a "bars" key if successful
    bars = data.get("bars", [])
    if not bars:
        print(f"[{symbol}] No data returned.")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(bars)
    # The bars typically include 't','o','h','l','c','v' for time,open,high,low,close,volume
    # Rename columns to a friendlier format
    df.rename(columns={
        "t": "timestamp", 
        "o": "open", 
        "h": "high", 
        "l": "low", 
        "c": "close", 
        "v": "volume",
        "n": "trades",
        "vw": "vwap"
    }, inplace=True)
    
    # Convert 'timestamp' to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Sort by time ascending
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def main():
    print(f"Fetching ~{YEARS_TO_FETCH} years of split-adjusted daily data from Alpaca...")
    print(f"Date range: {start_str} to {end_str}")
    
    for sym in ETF_SYMBOLS:
        try:
            print(f"---\nFetching {sym} ...")
            df = fetch_data(sym)
            if df is None or df.empty:
                print(f"{sym}: No data fetched.")
                continue
            
            # Save to CSV
            out_path = os.path.join(OUTPUT_DIR, f"{sym}.csv")
            # Reformat as typical columns: Date,Open,High,Low,Close,Volume
            df_to_save = df[["timestamp","open","high","low","close","volume"]].copy()
            df_to_save.rename(columns={"timestamp":"Date"}, inplace=True)
            
            # Convert to local time if you prefer, or keep it UTC
            # Example: df_to_save['Date'] = df_to_save['Date'].dt.tz_convert('US/Eastern')
            
            df_to_save.to_csv(out_path, index=False)
            print(f"{sym}: Saved {len(df_to_save)} rows to {out_path}")
            
            # Print price range to verify adjustment
            print(f"Price range: ${df_to_save['low'].min():.2f} - ${df_to_save['high'].max():.2f}")
            
        except Exception as e:
            print(f"Error fetching {sym}: {e}")

if __name__ == "__main__":
    main()
