from binance.client import Client
import pandas as pd
import os
from config import API_KEY, API_SECRET


client = Client(API_KEY, API_SECRET, tld='us')


symbol = "ETHUSDT"
interval = Client.KLINE_INTERVAL_5MINUTE

# Examples of date strings recognized by python-binance (parsing is done internally).
start_str = "25 Jun, 2024"  
end_str   = "25 Dec, 2024"  


klines = client.get_historical_klines(
    symbol=symbol,
    interval=interval,
    start_str=start_str,
    end_str=end_str
)


df = pd.DataFrame(klines, columns=[
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore"
])

# Convert numeric fields from strings to floats
df["open"] = df["open"].astype(float)
df["high"] = df["high"].astype(float)
df["low"] = df["low"].astype(float)
df["close"] = df["close"].astype(float)
df["volume"] = df["volume"].astype(float)

# Convert timestamps from milliseconds to datetime
df["open_time"] = pd.to_datetime(df["open_time"], unit='ms')
df["close_time"] = pd.to_datetime(df["close_time"], unit='ms')

# If you don't need columns like quote_asset_volume or taker volumes, you can drop them:
# df.drop(["quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"], axis=1, inplace=True)

print(df.head())

csv_filename = f"{symbol}_{interval}_historical.csv"
df.to_csv(csv_filename, index=False)
print(f"Saved to {csv_filename}")
