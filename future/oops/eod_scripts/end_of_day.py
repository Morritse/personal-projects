import os
import pandas as pd
import numpy as np

# 1) Import your config (contains FUTURES_SYMBOLS and BEST_PARAM_SET)
import config

# 2) Import the RefinedFuturesStrategy
from strategy import RefinedFuturesStrategy  # adjust import if needed

def load_data(symbol: str) -> pd.DataFrame:
    """
    Load daily data from 'live_data/<symbol>.csv'.
    Expects columns: date, open, high, low, close, volume...
    Then renames them to match 'Open','High','Low','Close','Volume'.
    """
    file_path = os.path.join("live_data", f"{symbol}.csv")
    if not os.path.exists(file_path):
        return pd.DataFrame()  # return empty if missing

    df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
    # Rename columns if they exist
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    }
    df.rename(columns=rename_map, inplace=True, errors="ignore")

    df.sort_index(inplace=True)
    return df

def main():
    # 1) Instantiate the RefinedFuturesStrategy using BEST_PARAM_SET
    strategy_params = config.BEST_PARAM_SET
    strategy = RefinedFuturesStrategy(**strategy_params)

    all_results = []

    # 2) Loop over each symbol in FUTURES_SYMBOLS
    for symbol in config.FUTURES_SYMBOLS.keys():
        df_raw = load_data(symbol)
        if df_raw.empty:
            print(f"[Warning] No data for {symbol} in live_data/; skipping.")
            continue

        # 3) Run single-instrument logic (which calculates signals, position, returns, etc.)
        df_out, metrics = strategy.run_single_instrument(df_raw)

        # 4) Grab the final row for today's EOD
        last_row = df_out.iloc[-1]
        date_str = last_row.name.strftime("%Y-%m-%d")

        # 5) Build a summary record
        record = {
            "symbol": symbol,
            "date": date_str,
            "close": last_row["Close"],
            "signal": last_row["signal"],           # final signal (-1 to +1)
            "position": last_row["position"],       # final position after risk mgmt
        }
        all_results.append(record)

    # 6) Convert all results into a DataFrame for EOD signals
    eod_signals = pd.DataFrame(all_results)
    eod_signals.sort_values(by="symbol", inplace=True)

    # Save signals to CSV
    signals_file = "signals/eod_signals.csv"
    os.makedirs("signals", exist_ok=True)
    eod_signals.to_csv(signals_file, index=False)

    # Print signals
    print("\n=== EOD Signals ===")
    print(eod_signals.to_string(index=False))

if __name__ == "__main__":
    main()
