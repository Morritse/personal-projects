# pipelines/data_pipeline.py

import os
import pandas as pd
from alpaca_trade_api.rest import REST, TimeFrame
from momentum_ai_trading.utils.config import ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL, RAW_DATA_PATH

api = REST(ALPACA_API_KEY, ALPACA_API_SECRET, BASE_URL)

def fetch_historical_data(symbol, start_date, end_date):
    """Fetch historical data for a given symbol."""
    bars = api.get_bars(symbol, TimeFrame.Day, start=start_date, end=end_date).df
    file_path = os.path.join(RAW_DATA_PATH, f"{symbol}_raw.csv")
    bars.to_csv(file_path)
    print(f"Saved raw data for {symbol} to {file_path}")

if __name__ == "__main__":
    fetch_historical_data("AAPL", "2020-01-01", "2023-01-01")
