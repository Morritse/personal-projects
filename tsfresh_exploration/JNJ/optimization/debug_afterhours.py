import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def debug_data(data):
    """Print debug information about the data."""
    print("\nData Structure:")
    for symbol in data.keys():
        df = data[symbol]
        print(f"\n{symbol}:")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nSample data:")
        print(df.head())
        print("\nValue ranges:")
        print(df.describe())

def analyze_spread(df):
    """Analyze the JNJ-XLV spread characteristics."""
    spread = df['jnj_returns_1H'] - df['xlv_returns_1H']
    
    print("\nSpread Analysis:")
    print("\nSpread statistics:")
    print(spread.describe())
    
    # Calculate z-scores
    rolling_mean = spread.rolling(window=20, min_periods=10).mean()
    rolling_std = spread.rolling(window=20, min_periods=10).std()
    zscore = (spread - rolling_mean) / rolling_std
    
    print("\nZ-score statistics:")
    print(zscore.describe())
    
    # Count potential signals
    divergences = abs(zscore) > 2.0
    print(f"\nPotential signals (|z-score| > 2): {divergences.sum()}")
    
    # Analyze by hour
    by_hour = pd.DataFrame({
        'spread': spread,
        'zscore': zscore,
        'hour': df.index.hour
    })
    
    print("\nSignals by hour:")
    for hour in range(16, 24):
        hour_data = by_hour[by_hour['hour'] == hour]
        signals = abs(hour_data['zscore']) > 2.0
        print(f"Hour {hour:02d}:00: {signals.sum()} signals")

def analyze_regimes(df):
    """Analyze regime classification."""
    print("\nRegime Analysis:")
    
    # Test different lookback windows
    windows = [10, 20, 30]
    vol_multipliers = [1.0, 1.5, 2.0]
    
    for window in windows:
        print(f"\nWindow = {window} hours:")
        xlv_ret = df['xlv_returns_1H'].rolling(window=window, min_periods=window//2).mean() * 252
        xlv_vol = df['xlv_returns_1H'].rolling(window=window, min_periods=window//2).std() * np.sqrt(252)
        
        for mult in vol_multipliers:
            vol_threshold = xlv_vol.median() * mult
            
            # Classify regimes
            regimes = pd.Series('normal', index=df.index)
            regimes[(xlv_ret > 0) & (xlv_vol <= vol_threshold)] = 'bull_low_vol'
            regimes[(xlv_ret > 0) & (xlv_vol > vol_threshold)] = 'bull_high_vol'
            regimes[(xlv_ret <= 0) & (xlv_vol <= vol_threshold)] = 'bear_low_vol'
            regimes[(xlv_ret <= 0) & (xlv_vol > vol_threshold)] = 'bear_high_vol'
            
            # Count regimes
            counts = regimes.value_counts()
            print(f"\nVol threshold = {mult}x median:")
            for regime, count in counts.items():
                print(f"{regime}: {count} ({count/len(df):.1%})")

def main():
    # Load data
    print("Loading market data...")
    cache_file = 'data/market_data.pkl'
    if not os.path.exists(cache_file):
        print("Please run analyze_xlv_correlation.py first to generate market data")
        return
    
    data = pd.read_pickle(cache_file)
    
    # Debug raw data
    debug_data(data)
    
    # Process data
    print("\nProcessing data...")
    
    # Create base DataFrame with JNJ data
    df = data['JNJ'].copy()
    df.index = df.index.tz_localize(None)
    df = df.resample('h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).ffill()
    
    # Add XLV data
    xlv_data = data['XLV'].copy()
    xlv_data.index = xlv_data.index.tz_localize(None)
    xlv_data = xlv_data.resample('h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).ffill()
    
    # Rename columns
    df = df.rename(columns={
        'close': 'jnj_close',
        'volume': 'jnj_volume'
    })
    df['xlv_close'] = xlv_data['close']
    df['xlv_volume'] = xlv_data['volume']
    
    # Calculate returns
    df['jnj_returns_1H'] = df['jnj_close'].pct_change()
    df['xlv_returns_1H'] = df['xlv_close'].pct_change()
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    # Filter for after-hours only
    df = df[df.index.hour >= 16]
    
    print("\nProcessed data shape:", df.shape)
    print("\nSample processed data:")
    print(df.head())
    print("\nValue ranges after processing:")
    print(df.describe())
    
    # Analyze spread characteristics
    analyze_spread(df)
    
    # Analyze regime classification
    analyze_regimes(df)

if __name__ == "__main__":
    main()
