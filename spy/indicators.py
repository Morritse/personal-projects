import pandas as pd
import numpy as np

###############################################################################
# Indicator Functions
###############################################################################

def compute_macd(df, fast=12, slow=26, signal=9, volume_weighted=False):
    """
    Compute MACD (12, 26, 9):
      - MACD_line = EMA(fast) - EMA(slow)
      - Signal_line = EMA(MACD_line)
      - Hist = MACD_line - Signal_line

    If volume_weighted=True, apply a (simple) volume-weighted logic:
      Weighted close = close * volume
      Then we do a volume-weighted EMA approximation.
    """
    if volume_weighted:
        df['weighted_close'] = df['close'] * df['volume']

        # Helper for volume-weighted EMA
        def vema(series, span):
            return series.ewm(span=span, adjust=False).mean()

        ema_fast = vema(df['weighted_close'], fast) / vema(df['volume'], fast)
        ema_slow = vema(df['weighted_close'], slow) / vema(df['volume'], slow)
    else:
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    df['MACD_line'] = ema_fast - ema_slow
    df['Signal_line'] = df['MACD_line'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD_line'] - df['Signal_line']

    return df

def compute_rsi(df, period=14, volume_weighted=False):
    """
    Compute RSI over 'period' bars.
    If volume_weighted=True, weigh price changes by volume (experimental).
    """
    delta = df['close'].diff()

    if volume_weighted:
        # Multiply each price change by volume
        delta = delta * df['volume']

    gain = (delta.where(delta > 0, 0)).abs()
    loss = (delta.where(delta < 0, 0)).abs()

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df

def compute_bollinger(df, period=20, std_dev=2):
    """
    Bollinger Bands: 
      Middle = SMA(period), 
      Upper = Middle + std_dev * rolling_std, 
      Lower = Middle - std_dev * rolling_std
    """
    df['BB_mid'] = df['close'].rolling(period).mean()
    df['BB_std'] = df['close'].rolling(period).std()

    df['BB_upper'] = df['BB_mid'] + std_dev * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - std_dev * df['BB_std']
    return df

def compute_stochastics(df, k_period=14, d_period=3, smooth=3):
    """
    Stochastics (Full):
      %K = 100 * (close - L14) / (H14 - L14)
      Then we smooth %K for 'smooth' bars,
      Then %D is a moving average of %K over d_period bars.
    """
    low_k = df['low'].rolling(k_period).min()
    high_k = df['high'].rolling(k_period).max()

    df['stoch_k_unsmoothed'] = 100 * (df['close'] - low_k) / (high_k - low_k)
    df['stoch_k'] = df['stoch_k_unsmoothed'].rolling(smooth).mean()
    df['stoch_d'] = df['stoch_k'].rolling(d_period).mean()

    return df

def compute_donchian(df, lookback=20):
    """
    Donchian Channels (20):
      donchian_high = highest high of last 20 bars
      donchian_low = lowest low of last 20 bars
    """
    df['donchian_high'] = df['high'].rolling(lookback).max()
    df['donchian_low'] = df['low'].rolling(lookback).min()
    return df

###############################################################################
# Main Script to Load, Enhance, and Save
###############################################################################

def add_indicators(df, volume_weighted=False):
    """
    Apply our suite of indicators to the given DataFrame in-place.
    Returns the DataFrame with new columns added.
    """
    df = compute_macd(df, volume_weighted=volume_weighted)
    df = compute_rsi(df, volume_weighted=volume_weighted)
    df = compute_bollinger(df)
    df = compute_stochastics(df)
    df = compute_donchian(df)

    # (Optional) Drop intermediate columns if you don't want them:
    # E.g., stoch_k_unsmoothed, BB_std, weighted_close
    df.drop(['stoch_k_unsmoothed','BB_std','weighted_close'], 
            axis=1, inplace=True, errors='ignore')
    
    return df

def main():
    # Filenames (adjust if needed)
    file_1m_cleaned = "spy_1m_cleaned.csv"
    file_5m_cleaned = "spy_5m_cleaned.csv"
    file_15m_cleaned = "spy_15m_cleaned.csv"
    
    file_1m_output = "spy_1m_with_indicators.csv"
    file_5m_output = "spy_5m_with_indicators.csv"
    file_15m_output = "spy_15m_with_indicators.csv"

    # 1) Load Cleaned CSVs
    df_1m = pd.read_csv(file_1m_cleaned, parse_dates=["timestamp"], index_col="timestamp")
    df_5m = pd.read_csv(file_5m_cleaned, parse_dates=["timestamp"], index_col="timestamp")
    df_15m = pd.read_csv(file_15m_cleaned, parse_dates=["timestamp"], index_col="timestamp")

    # 2) Add indicators. 
    # Choose volume_weighted=True if you want that experimental weighting
    df_1m = add_indicators(df_1m, volume_weighted=False)
    df_5m = add_indicators(df_5m, volume_weighted=False)
    df_15m = add_indicators(df_15m, volume_weighted=False)

    # 3) Save results
    df_1m.to_csv(file_1m_output)
    df_5m.to_csv(file_5m_output)
    df_15m.to_csv(file_15m_output)
    
    print("Done! Indicator-enhanced CSVs saved:" 
          f"\n  - {file_1m_output}"
          f"\n  - {file_5m_output}"
          f"\n  - {file_15m_output}")

if __name__ == "__main__":
    main()
