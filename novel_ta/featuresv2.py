import pandas as pd
import numpy as np
import talib  # pip install TA-Lib

def add_features(df, df_spy=None, dropna=True):
    """
    Given a DataFrame with columns: [date, open, high, low, close, volume],
    append technical features including correlation with SPY if provided.
    Returns a new DataFrame with additional columns.
    """
    # Ensure consistent column casing and sorting
    df = df.rename(columns=str.lower)
    df.sort_values("date", inplace=True)
    
    # Basic safety check
    required_cols = ['date','open','high','low','close','volume']
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    
    # Set date as index for calculations
    df.set_index("date", inplace=True)
    
    # 1) Returns
    df['daily_return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close']).diff()
    
    # 2) Future Returns (1-day and 5-day)
    # 1-day returns
    df['future_return_1d'] = df['close'].shift(-1) / df['close'] - 1
    df['target_1d'] = (df['future_return_1d'] > 0).astype(int)
    # 5-day returns
    df['future_return_5d'] = (df['close'].shift(-5) - df['close']) / df['close']
    df['target_5d'] = (df['future_return_5d'] > 0).astype(int)
    
    # 3) Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # 4) Bollinger Bands (20-day, 2 stdev)
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['bb_upper'] = upper
    df['bb_middle'] = middle
    df['bb_lower'] = lower
    
    # 5) RSI (7-day and 14-day)
    df['rsi_7'] = talib.RSI(df['close'], timeperiod=7)
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    
    # 6) ATR (14-day)
    df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    
    # 7) Volume Features
    df['vol_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # 8) Optional: Correlation with SPY
    if df_spy is not None:
        # Prepare SPY data
        df_spy = df_spy.rename(columns=str.lower)
        df_spy.sort_values("date", inplace=True)
        df_spy.set_index("date", inplace=True)
        
        # Calculate returns for both
        df_spy['ret_spy'] = df_spy['close'].pct_change()
        
        # Merge on index (date)
        df = df.join(df_spy[['close', 'ret_spy']], rsuffix='_spy')
        
        # Calculate 20-day rolling correlation
        df['corr_20_spy'] = df['daily_return'].rolling(20).corr(df['ret_spy'])
        
        # Clean up
        df.drop(['close_spy', 'ret_spy'], axis=1, inplace=True)
    
    # Reset index to make date a column again
    df.reset_index(inplace=True)
    
    # Drop NaNs if requested
    if dropna:
        df.dropna(inplace=True)
        
    return df

def process_instrument_csv(filepath, spy_filepath=None):
    """
    Reads a CSV, adds features, and returns the processed DataFrame.
    If spy_filepath is provided, includes correlation features with SPY.
    """
    # Load instrument data
    df = pd.read_csv(filepath, names=['date','open','high','low','close','volume'], skiprows=1)
    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)  # Remove timezone info
    
    # Load SPY data if provided
    df_spy = None
    if spy_filepath:
        df_spy = pd.read_csv(spy_filepath, names=['date','open','high','low','close','volume'], skiprows=1)
        df_spy['date'] = pd.to_datetime(df_spy['date']).dt.tz_localize(None)  # Remove timezone info
    
    # Compute features
    df = add_features(df, df_spy)
    
    return df

if __name__ == "__main__":
    instruments = ["DBC", "DIA", "GLD", "IEF", "IWM", "LQD", "QQQ", "SLV", 
                  "SPY", "TLT", "UNG", "USO", "UUP", "XLE", "XLF"]
    
    # Process SPY first since we'll need it for correlations
    spy_filepath = "alpaca_5yr_data/SPY_5yr_daily.csv"
    print("Processing SPY...")
    df_spy = pd.read_csv(spy_filepath, names=['date','open','high','low','close','volume'], skiprows=1)
    df_spy['date'] = pd.to_datetime(df_spy['date']).dt.tz_localize(None)  # Remove timezone info
    
    for symbol in instruments:
        if symbol == "SPY":  # Skip SPY since we already processed it
            continue
            
        input_filepath = f"alpaca_5yr_data/{symbol}_5yr_daily.csv"
        output_filepath = f"featuresv2_data/{symbol}_with_features_v2_1d.csv"
        print(f"Processing {symbol}...")
        
        try:
            df_processed = process_instrument_csv(input_filepath, spy_filepath)
            df_processed.to_csv(output_filepath, index=False)
            print(f"Saved features for {symbol} to {output_filepath}")
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    # Process SPY separately without correlation to itself
    print("\nProcessing SPY without self-correlation...")
    df_spy_processed = process_instrument_csv(spy_filepath)
    df_spy_processed.to_csv("featuresv2_data/SPY_with_features_v2_1d.csv", index=False)
    print("Completed processing all instruments.")
