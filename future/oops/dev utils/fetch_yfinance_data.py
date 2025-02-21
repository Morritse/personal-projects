#!/usr/bin/env python3

import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Map our symbols to Yahoo Finance futures symbols
FUTURES = {
    # Equity Index
    "ES": "ES=F",  # E-mini S&P 500
    "NQ": "NQ=F",  # E-mini NASDAQ 100
    "YM": "YM=F",  # E-mini Dow
    "RTY": "RTY=F", # E-mini Russell 4000
    
    # Interest Rates
    "ZB": "ZB=F",  # 30-Year T-Bond
    "ZN": "ZN=F",  # 10-Year T-Note
    "ZF": "ZF=F",  # 5-Year T-Note
    
    # Energy
    "CL": "CL=F",  # Crude Oil
    "NG": "NG=F",  # Natural Gas
    "RB": "RB=F",  # RBOB Gasoline
    "HO": "HO=F",  # Heating Oil
    
    # Metals
    "GC": "GC=F",  # Gold
    "SI": "SI=F",  # Silver
    "HG": "HG=F",  # Copper
    
    # Agriculture
    "ZC": "ZC=F",  # Corn
    "ZW": "ZW=F",  # Wheat
    "ZS": "ZS=F",  # Soybeans
    
    # Livestock
    "LE": "LE=F",  # Live Cattle
}

def fetch_400_bars(symbol, yf_symbol):
    """Fetch last 400 daily bars for a futures contract."""
    print(f"[{symbol}] Fetching data for {yf_symbol}...")
    
    try:
        # Request more than 400 days to account for weekends/holidays
        end_date = datetime.now()
        start_date = end_date - timedelta(days=500)
        
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            print(f"[{symbol}] No data returned")
            return None
            
        # Rename columns to match our format
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        # Add required columns
        df['average'] = 0  # Placeholder
        df['barCount'] = 0  # Placeholder
        df['localSymbol'] = yf_symbol
        df['expiry'] = ''  # Yahoo doesn't provide expiry info for continuous contracts
        
        # Keep only required columns
        df = df[['open', 'high', 'low', 'close', 'volume', 'average', 'barCount', 'localSymbol', 'expiry']]
        
        # Take last 400 bars
        if len(df) > 400:
            df = df.iloc[-400:]
            
        return df
        
    except Exception as ex:
        print(f"[{symbol}] Error: {ex}")
        return None

def main():
    # Create output directory
    out_folder = 'base_data'
    os.makedirs(out_folder, exist_ok=True)
    
    # Process each symbol
    for symbol, yf_symbol in FUTURES.items():
        df = fetch_400_bars(symbol, yf_symbol)
        
        if df is not None and not df.empty:
            out_path = os.path.join(out_folder, f"{symbol}_400D.csv")
            df.to_csv(out_path)
            print(f"[{symbol}] Saved {len(df)} bars to {out_path}")
        else:
            print(f"[{symbol}] Failed to fetch data")
            
    print("\nAll done!")

if __name__ == '__main__':
    main()
