#!/usr/bin/env python3

import os
import logging
from datetime import datetime
import pandas as pd
from ib_insync import IB, Future, ContFuture, util

# Replace with all your desired futures
FUTURES = {
    # Equity Index (CME / CBOT)
    "ES": {"exchange": "CME",   "name": "E-mini S&P 500", "use_cont": False},
    "NQ": {"exchange": "CME",   "name": "E-mini Nasdaq 100", "use_cont": False},
    "YM": {"exchange": "CBOT",  "name": "E-mini Dow", "use_cont": False},
    "RTY": {"exchange": "CME",  "name": "E-mini Russell 2000", "use_cont": False},
    
    # Interest Rates (CBOT) - Using continuous for these
    "ZB": {"exchange": "CBOT",  "name": "30-Year Treasury Bond", "use_cont": True},
    "ZN": {"exchange": "CBOT",  "name": "10-Year Treasury Note", "use_cont": True},
    "ZF": {"exchange": "CBOT",  "name": "5-Year Treasury Note", "use_cont": True},
    
    # Energy (NYMEX)
    "CL": {"exchange": "NYMEX", "name": "Crude Oil (WTI)", "use_cont": False},
    "NG": {"exchange": "NYMEX", "name": "Natural Gas", "use_cont": False},
    "RB": {"exchange": "NYMEX", "name": "RBOB Gasoline", "use_cont": False},
    "HO": {"exchange": "NYMEX", "name": "Heating Oil", "use_cont": False},
    
    # Metals (COMEX) - Using continuous for Gold
    "GC": {"exchange": "COMEX", "name": "Gold", "use_cont": True},
    "SI": {"exchange": "COMEX", "name": "Silver", "use_cont": False},
    "HG": {"exchange": "COMEX", "name": "Copper", "use_cont": False},

    # Agriculture (CBOT)
    "ZC": {"exchange": "CBOT",  "name": "Corn", "use_cont": False},
    "ZW": {"exchange": "CBOT",  "name": "Wheat", "use_cont": False},
    "ZS": {"exchange": "CBOT",  "name": "Soybeans", "use_cont": False},

    # Livestock (CME)
    "LE": {"exchange": "CME",   "name": "Live Cattle", "use_cont": False},
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

def fetch_continuous_200_bars(ib, symbol, exchange):
    """
    Fetch ~200 daily bars using a continuous futures contract.
    """
    try:
        # Create continuous futures contract
        contract = ContFuture(symbol=symbol, exchange=exchange, currency='USD')
        
        # Qualify the contract
        contract = ib.qualifyContracts(contract)[0]
        
        print(f"[{symbol}] Using continuous contract")
        
        # Request historical data
        bars = ib.reqHistoricalData(
            contract=contract,
            endDateTime='',
            durationStr='1 Y',  # Request 1 year of data to ensure we get enough bars
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        
        if not bars:
            print(f"[{symbol}] No bars returned from IB.")
            return None
        
        # Convert to DataFrame
        df = util.df(bars)
        if df.empty:
            print(f"[{symbol}] DataFrame is empty.")
            return None

        df.set_index('date', inplace=True)
        df = df[['open','high','low','close','volume']].copy()
        df.sort_index(inplace=True)
        
        # Take the last 200 bars if we have more
        if len(df) > 200:
            df = df.iloc[-200:]
            
        df['localSymbol'] = 'CONTINUOUS'
        df['expiry'] = ''
        
        return df
        
    except Exception as ex:
        print(f"[{symbol}] Error fetching continuous data: {ex}")
        return None

def fetch_200_bars(ib, symbol, exchange):
    """
    Fetch ~200 daily bars for the front-month contract of `symbol`.
    Returns a DataFrame with daily OHLC + volume if successful.
    """
    # 1) Resolve the front-month contract details
    base_contract = Future(symbol=symbol, exchange=exchange, currency='USD')
    cdetails = ib.reqContractDetails(base_contract)
    if not cdetails:
        print(f"[{symbol}] No contract details found!")
        return None

    # Filter by matching exchange and sort
    valid_contracts = [
        cd.contract for cd in cdetails if cd.contract.exchange == exchange
    ]
    if not valid_contracts:
        print(f"[{symbol}] No valid contracts for exchange={exchange}.")
        return None

    valid_contracts.sort(key=parse_expiry)
    
    # pick first that hasn't expired
    now = datetime.now()
    front_contract = None
    for con in valid_contracts:
        if parse_expiry(con) > now:
            front_contract = con
            break
    if not front_contract:
        # fallback if all are expired
        front_contract = valid_contracts[-1]

    print(f"[{symbol}] Using front-month localSymbol={front_contract.localSymbol}, "
          f"expiry={front_contract.lastTradeDateOrContractMonth}")

    # 2) Request 200 daily bars
    try:
        bars = ib.reqHistoricalData(
            contract=front_contract,
            endDateTime='',
            durationStr='1 Y',       # request 1 year to ensure we get enough bars
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )
        if not bars:
            print(f"[{symbol}] No bars returned from IB.")
            return None
        
        # Convert to DataFrame
        df = util.df(bars)
        if df.empty:
            print(f"[{symbol}] DataFrame is empty.")
            return None

        df.set_index('date', inplace=True)
        df = df[['open','high','low','close','volume']].copy()
        df.sort_index(inplace=True)
        
        # Take the last 200 bars if we have more
        if len(df) > 200:
            df = df.iloc[-200:]
            
        df['localSymbol'] = front_contract.localSymbol
        df['expiry']      = front_contract.lastTradeDateOrContractMonth
        
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
            
            # Use continuous contract for specified symbols
            if info['use_cont']:
                df = fetch_continuous_200_bars(ib, symbol, exch)
            else:
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
