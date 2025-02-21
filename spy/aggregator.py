import pandas as pd

# -- CONFIG --
filename = "spy_1m.csv"  # Adjust to your raw file
output_1m = "spy_1m_cleaned.csv"
output_5m = "spy_5m_cleaned.csv"
output_15m = "spy_15m_cleaned.csv"

# 1) Read CSV, parse timestamps, set index to 'timestamp'
df_1m = pd.read_csv(
    filename,
    parse_dates=["timestamp"],
    index_col="timestamp"
)

# 2) Drop the last two columns by position 
#    (If they are named columns, you could do df_1m.drop(['trade_count','vwap'], axis=1))
df_1m.drop(df_1m.columns[-2:], axis=1, inplace=True)

# 3) Remove non-trading rows (example heuristics):
#    a) Drop rows where open/high/low/close are NaN or zero
df_1m.dropna(subset=["open", "high", "low", "close"], how="any", inplace=True)
df_1m = df_1m[(df_1m["open"] != 0) &
              (df_1m["high"] != 0) &
              (df_1m["low"] != 0) &
              (df_1m["close"] != 0)]

#    b) Remove weekends/holidays by day-of-week: 0=Monday,...,4=Friday
df_1m = df_1m[df_1m.index.dayofweek < 5]

# (Optional) If you also want to remove rows with volume=0 (assuming no trades):
# df_1m = df_1m[df_1m["volume"] != 0]

df_1m = df_1m.tz_convert("America/New_York")

#    B) Keep only 09:30–16:00 local time
df_1m = df_1m.between_time("09:30", "16:00")
# Now df_1m should be "clean" 1-minute data (no extra columns, no obvious non-trading).
# Save it
df_1m.to_csv(output_1m)

# 4) Resample to 5-minute bars
df_5m = df_1m.resample("5T").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
})

# Remove any fully empty bars (if they exist)
df_5m.dropna(subset=["open","high","low","close"], how="all", inplace=True)
df_5m.to_csv(output_5m)

# 5) Resample to 15-minute bars
df_15m = df_1m.resample("15T").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
})

df_15m.dropna(subset=["open","high","low","close"], how="all", inplace=True)
df_15m.to_csv(output_15m)

print("Done! Cleaned 1m, 5m, and 15m files saved.")
