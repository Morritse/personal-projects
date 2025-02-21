#!/usr/bin/env python3

import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from ib_insync import IB, Future, util

# Replace with all your desired futures
FUTURES = {
    # Equity Index (CME / CBOT)
    "ES": {"exchange": "CME",   "name": "E-mini S&P 500"},
    "NQ": {"exchange": "CME",   "name": "E-mini Nasdaq 100"},
    "YM": {"exchange": "CBOT",  "name": "E-mini Dow"},
    "RTY": {"exchange": "CME",  "name": "E-mini Russell 2000"},
    
    # Interest Rates (CBOT)
    "ZB": {"exchange": "CBOT",  "name": "30-Year Treasury Bond"},
    "ZN": {"exchange": "CBOT",  "name": "10-Year Treasury Note"},
    "ZF": {"exchange": "CBOT",  "name": "5-Year Treasury Note"},
    
    # Energy (NYMEX)
    "CL": {"exchange": "NYMEX", "name": "Crude Oil (WTI)"},
    "NG": {"exchange": "NYMEX", "name": "Natural Gas"},
    "RB": {"exchange": "NYMEX", "name": "RBOB Gasoline"},
    "HO": {"exchange": "NYMEX", "name": "Heating Oil"},
    
    # Metals (COMEX)
    "GC": {"exchange": "COMEX", "name": "Gold"},
    "SI": {"exchange": "COMEX", "name": "Silver"},
    "HG": {"exchange": "COMEX", "name": "Copper"},

    # Agriculture (CBOT)
    "ZC": {"exchange": "CBOT",  "name": "Corn"},
    "ZW": {"exchange": "CBOT",  "name": "Wheat"},
    "ZS": {"exchange": "CBOT",  "name": "Soybeans"},

    # Livestock (CME)
    "LE": {"exchange": "CME",   "name": "Live Cattle"},
}

def connect_ib():
    """Connect to IB Gateway or TWS"""
    ib = IB()
    try:
        print("Connecting to IB on 127.0.0.1:7497...")
        ib.connect(host='127.0.0.1', port=7497, clientId=1, readonly=True)
        print("Connected.")
    except Exception as e:
        print("Failed to connect:", e)
        return None
    return ib

def parse_expiry(contract):
    """Parse contract.lastTradeDateOrContractMonth into a datetime."""
    expiry_str = contract.lastTradeDateOrContractMonth
    # if format is YYYYMM
    if len(expiry_str) == 6:
        expiry_str += '01'
    return datetime.strptime(expiry_str, '%Y%m%d')

def fetch_200_bars(ib, symbol, exchange):
    """
    Fetch 200 daily bars by trying multiple contracts if needed.
    Returns a DataFrame with daily OHLC + volume if successful.
    """
    # Get all available contracts
    base_contract = Future(symbol=symbol, exchange=exchange, currency='USD')
    cdetails = ib.reqContractDetails(base_contract)
    if not cdetails:
        print(f"[{symbol}] No contract details found!")
        return None

    # Filter by matching exchange and sort by expiry
    valid_contracts = [
        cd.contract for cd in cdetails if cd.contract.exchange == exchange
    ]
    if not valid_contracts:
        print(f"[{symbol}] No valid contracts for exchange={exchange}.")
        return None

    valid_contracts.sort(key=parse_expiry)
    
    # Find active contract
    now = datetime.now()
    active_contract = None
    for con in valid_contracts:
        if parse_expiry(con) > now:
            active_contract = con
            break
    if not active_contract:
        active_contract = valid_contracts[-1]

    print(f"[{symbol}] Using contract {active_contract.localSymbol}")
    
    # Try to get data from active contract first
    try:
        bars = ib.reqHistoricalData(
            contract=active_contract,
            endDateTime='',
            durationStr='1 Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if not bars:
            print(f"[{symbol}] No bars returned.")
            return None
            
        df = util.df(bars)
        if df.empty:
            print(f"[{symbol}] DataFrame is empty.")
            return None

        df.set_index('date', inplace=True)
        df = df[['open','high','low','close','volume']].copy()
        df.sort_index(inplace=True)
        
        # If we don't have enough bars, try previous contract
        if len(df) < 200 and len(valid_contracts) > 1:
            prev_idx = valid_contracts.index(active_contract) - 1
            if prev_idx >= 0:
                prev_contract = valid_contracts[prev_idx]
                print(f"[{symbol}] Also trying previous contract {prev_contract.localSymbol}")
                
                prev_bars = ib.reqHistoricalData(
                    contract=prev_contract,
                    endDateTime=df.index[0].strftime('%Y%m%d'),  # Start where current data ends
                    durationStr='6 M',
                    barSizeSetting='1 day',
                    whatToShow='TRADES',
                    useRTH=True,
                    formatDate=1
                )
                
                if prev_bars:
                    prev_df = util.df(prev_bars)
                    prev_df.set_index('date', inplace=True)
                    prev_df = prev_df[['open','high','low','close','volume']].copy()
                    
                    # Combine data
                    df = pd.concat([prev_df, df])
                    df = df[~df.index.duplicated(keep='last')]  # Remove any overlaps
                    df.sort_index(inplace=True)
        
        # Take last 200 bars
        if len(df) > 200:
            df = df.iloc[-200:]
            
        df['localSymbol'] = active_contract.localSymbol
        df['expiry'] = active_contract.lastTradeDateOrContractMonth
        
        return df
        
    except Exception as ex:
        print(f"[{symbol}] Error fetching data: {ex}")
        return None

def main():
    # Create base_data folder
    out_folder = 'base_data'
    os.makedirs(out_folder, exist_ok=True)

    ib = connect_ib()
    if not ib:
        return
    
    try:
        for symbol, info in FUTURES.items():
            exch = info['exchange']
            df = fetch_200_bars(ib, symbol, exch)
                
            if df is not None and not df.empty:
                out_path = os.path.join(out_folder, f"{symbol}_200D.csv")
                df.to_csv(out_path)
                print(f"[{symbol}] Fetched {len(df)} bars, saved to {out_path}")
            else:
                print(f"[{symbol}] No data fetched.")
    finally:
        ib.disconnect()
        print("Disconnected from IB.")

if __name__ == '__main__':
    util.logToConsole(logging.ERROR)
    main()
