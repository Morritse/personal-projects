#!/usr/bin/env python3

import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from ib_insync import IB, Future, util

# Import the config file
from config import FUTURES_SYMBOLS

def connect_to_ib():
    """
    Connect to local IB Gateway/TWS (paper or live) on the default port 7497.
    Set `readonly=False` if you plan to place trades in the same session.
    """
    ib = IB()
    try:
        print("Connecting to IB on 127.0.0.1:7497...")
        ib.connect(host='127.0.0.1', port=7497, clientId=1, readonly=True)
        print("Connection successful.")
    except Exception as e:
        print(f"IB connection failed: {e}")
        return None
    return ib

def get_front_month_contract(ib, symbol, exchange):
    """
    Query all contract details for the given symbol+exchange,
    then pick the front-month (nearest future with expiry > now).
    """
    base_contract = Future(symbol=symbol, exchange=exchange, currency='USD')
    details_list = ib.reqContractDetails(base_contract)
    if not details_list:
        print(f"No contract details found for {symbol} on {exchange}")
        return None

    valid_contracts = []
    now = datetime.now()
    for cd in details_list:
        c = cd.contract
        if c.exchange == exchange and c.lastTradeDateOrContractMonth:
            expiry_str = c.lastTradeDateOrContractMonth
            # For example "20250321" or "202503"
            if len(expiry_str) == 6:  
                expiry_str += "01"  # interpret YYYYMM as YYYYMM01
            try:
                expiry_dt = datetime.strptime(expiry_str, "%Y%m%d")
                if expiry_dt > now:
                    valid_contracts.append((expiry_dt, c))
            except ValueError:
                continue

    valid_contracts.sort(key=lambda x: x[0])  # sort by expiry ascending
    if not valid_contracts:
        print(f"No valid (unexpired) futures found for {symbol}")
        return None

    # The nearest unexpired contract
    front_expiry, front_contract = valid_contracts[0]
    return front_contract

def fetch_eod_bar(ib, contract, days_back=1):
    """
    Fetch daily bar(s) for the contract.
    - endDateTime uses local time: "YYYYMMDD 23:59:59"
    - `days_back` can be 1 or 2 to ensure we get today's bar.
    """
    end_str = datetime.now().strftime('%Y%m%d 23:59:59')
    duration_str = f"{days_back} D" if days_back > 1 else "1 D"

    print(f"Requesting daily bars for {contract.localSymbol} until {end_str}, duration={duration_str}")
    bars = ib.reqHistoricalData(
        contract=contract,
        endDateTime=end_str,
        durationStr=duration_str,
        barSizeSetting="1 day",
        whatToShow="TRADES",  # or MIDPOINT
        useRTH=False,
        formatDate=1,
        keepUpToDate=False
    )

    if bars:
        df = util.df(bars)
        df.set_index('date', inplace=True)
        return df
    else:
        print(f"No data returned for {contract.localSymbol}")
        return None

def main():
    # Create data folder if not present
    os.makedirs("daily_data", exist_ok=True)

    ib = connect_to_ib()
    if not ib:
        return

    # We'll store the final snapshot in memory
    all_latest_rows = []

    # Loop over the FUTURES_SYMBOLS from config
    for symbol, info in FUTURES_SYMBOLS.items():
        exchange = info["exchange"]
        print("\n----------------------------")
        print(f"Fetching EOD data for {symbol} ({info['name']}) on {exchange}...")
        front_contract = get_front_month_contract(ib, symbol, exchange)
        if not front_contract:
            continue

        # Fetch last 2 days of daily bars, to ensure we have today's bar
        df = fetch_eod_bar(ib, front_contract, days_back=2)
        if df is not None and not df.empty:
            # Keep the last (most recent) bar
            latest_date = df.index[-1]
            row_dict = df.iloc[-1].to_dict()
            row_dict["symbol"] = symbol
            row_dict["localSymbol"] = front_contract.localSymbol
            row_dict["expiry"] = front_contract.lastTradeDateOrContractMonth
            row_dict["date"] = latest_date

            all_latest_rows.append(row_dict)

            # Optionally, save the entire small DataFrame
            outfile = f"daily_data/{symbol}_{front_contract.localSymbol}.csv"
            df.to_csv(outfile)
            print(f"Saved daily bars to {outfile}")

    # Combine final snapshot
    if all_latest_rows:
        df_snapshot = pd.DataFrame(all_latest_rows)
        df_snapshot.set_index("symbol", inplace=True)
        snapshot_file = "daily_data/futures_eod_snapshot.csv"
        df_snapshot.to_csv(snapshot_file)
        print(f"\nEOD snapshot saved to {snapshot_file}:\n{df_snapshot}\n")
    else:
        print("No data to snapshot.")

    ib.disconnect()
    print("\nDisconnected from IB.")

if __name__ == "__main__":
    util.logToConsole(logging.ERROR)
    main()
