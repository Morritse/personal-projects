import os
import logging
from ib_insync import IB, util, ContFuture
import pandas as pd

# Your futures definitions
FUTURES = {
    # Energy (NYMEX)
    'CL': {'exchange': 'NYMEX', 'name': 'Crude Oil'},
    'NG': {'exchange': 'NYMEX', 'name': 'Natural Gas'},

    # Metals (COMEX)
    'GC': {'exchange': 'COMEX', 'name': 'Gold'},
    'SI': {'exchange': 'COMEX', 'name': 'Silver'},
    'HG': {'exchange': 'COMEX', 'name': 'Copper'},

    # Equity Index (CME)
    'ES': {'exchange': 'CME', 'name': 'E-mini S&P 500'},
    'NQ': {'exchange': 'CME', 'name': 'E-mini NASDAQ'},
    'RTY': {'exchange': 'CME', 'name': 'E-mini Russell 2000'},

    # Equity Index (CBOT)
    'YM': {'exchange': 'CBOT', 'name': 'E-mini Dow'},

    # Rates (CBOT)
    'ZN': {'exchange': 'CBOT', 'name': '10-Year Treasury'},
    'ZF': {'exchange': 'CBOT', 'name': '5-Year Treasury'},
    'ZB': {'exchange': 'CBOT', 'name': '30-Year Treasury'},

    # Agriculture (CBOT)
    'ZC': {'exchange': 'CBOT', 'name': 'Corn'},
    'ZW': {'exchange': 'CBOT', 'name': 'Wheat'},
    'ZS': {'exchange': 'CBOT', 'name': 'Soybeans'},
}

def main():
    # 1) Connect to IB
    ib = IB()
    try:
        print("Connecting to IB TWS/Gateway...")
        ib.connect('127.0.0.1', 7497, clientId=999)  # or whatever host/port/clientId
        print("Connected.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # 2) Create a folder for CSV files
    os.makedirs('data', exist_ok=True)

    # 3) For each symbol, request 5 years of daily data using a ContFuture
    for symbol, info in FUTURES.items():
        print(f"\n==> Fetching 5 years daily for {symbol} ({info['name']})")

        # IB’s built-in continuous futures contract
        cont = ContFuture(symbol=symbol, exchange=info['exchange'], currency='USD')
        
        # Resolve the contract
        cont_details = ib.qualifyContracts(cont)
        if not cont_details:
            print(f"  Could not qualify ContFuture for {symbol}. Skipping.")
            continue

        # 4) Request 5 years of daily bars
        try:
            bars = ib.reqHistoricalData(
                cont,
                endDateTime='',            # or e.g. '20241229 23:59:59 US/Eastern'
                durationStr='5 Y',        # 5 years
                barSizeSetting='1 day',
                whatToShow='TRADES',      # or 'MIDPOINT' or 'BID_ASK' or 'ADJUSTED_LAST'
                useRTH=False,             # set to True if you only want regular trading hours
                formatDate=1,
                keepUpToDate=False
            )
            if not bars:
                print(f"  No data returned for {symbol}.")
                continue

            # Convert to pandas
            df = util.df(bars)
            df.set_index('date', inplace=True)
            # Optionally rename columns or keep as is
            df.sort_index(inplace=True)

            # 5) Save to CSV
            out_name = f"data/{symbol}_cont_future_daily.csv"
            df.to_csv(out_name)
            print(f"  Saved {len(df)} rows -> {out_name}")

        except Exception as err:
            print(f"  Error fetching {symbol}: {err}")

    # Disconnect
    ib.disconnect()
    print("\nDone. Disconnected from IB.")

if __name__ == '__main__':
    # Lower log level if you want fewer messages
    util.logToConsole(logging.ERROR)
    main()
