import pandas as pd

df = pd.read_csv('historical_BTC.csv')
print(df.head())


# Convert open_time and close_time to datetime
df['open_time'] = pd.to_datetime(df['open_time'])
df['close_time'] = pd.to_datetime(df['close_time'])

# Convert numeric columns from strings to floats
float_cols = ['open', 'high', 'low', 'close', 'volume',
              'quote_asset_volume',
              'taker_buy_base_asset_volume',
              'taker_buy_quote_asset_volume']

for col in float_cols:
    df[col] = df[col].astype(float)

# Verify
print(df.dtypes)


print(df['open_time'].min(), df['open_time'].max())


df['time_diff'] = df['open_time'].diff()
print(df['time_diff'].value_counts())


print(df[['open', 'high', 'low', 'close']].describe())
