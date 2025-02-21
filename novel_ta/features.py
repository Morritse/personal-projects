import pandas as pd
import numpy as np
import talib  # pip install TA-Lib

def add_features(df):
    """
    Given a DataFrame with columns: [date, open, high, low, close, volume],
    append some common technical features.
    Returns a new DataFrame with additional columns.
    """
    # Make sure df is sorted by date
    df = df.sort_values("date")
    df.set_index("date", inplace=True)  # optional if you want date as index
    
    # 1) Daily Return
    # (close - close_prev) / close_prev
    df['daily_return'] = df['close'].pct_change()
    
    # 2) Log Return
    df['log_return'] = np.log(df['close']).diff()
    
    # 3) Rolling Mean/EMA
    # Example: 20-day simple moving average
    df['sma_20'] = df['close'].rolling(window=20).mean()
    # 50-day exponential moving average
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 4) Bollinger Bands (using ta-lib for convenience)
    #   We'll compute a 20-day BB with 2 stdev
    #   TA-Lib function: BBANDS -> returns upperband, middleband, lowerband
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    
    # 5) RSI (14-day)
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    
    # 6) ATR (14-day)
    # ATR uses high, low, close
    df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 7) Volume-based feature (e.g., volume / rolling avg volume)
    df['vol_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Drop initial NaNs from rolling calculations (optional)
    df.dropna(inplace=True)
    return df

def process_instrument_csv(filepath):
    """
    Reads a CSV, adds features, and returns the processed DataFrame.
    """
    # Load CSV
    df = pd.read_csv(filepath, parse_dates=['Date'])
    # Rename columns to standard naming if needed
    df.columns = ['date','open','high','low','close','volume']
    
    # Compute features
    df = add_features(df)
    
    return df

# Example usage
if __name__ == "__main__":
    instruments = ["DBC", "DIA", "GLD", "IEF", "IWM", "LQD", "QQQ", "SLV", 
                  "SPY", "TLT", "UNG", "USO", "UUP", "XLE", "XLF"]
    
    for symbol in instruments:
        input_filepath = f"alpaca_5yr_data/{symbol}_5yr_daily.csv"
        output_filepath = f"alpaca_5yr_data/{symbol}_with_features.csv"
        print(f"Processing {symbol}...")
        df_processed = process_instrument_csv(input_filepath)
        df_processed.to_csv(output_filepath)
        print(f"Saved features for {symbol} to {output_filepath}")
