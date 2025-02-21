import numpy as np
import pandas as pd

def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data: Price series data
        period: RSI period (default: 14)
    
    Returns:
        RSI values as pandas Series
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> tuple:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data: Price series data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
    
    Returns:
        Tuple of (MACD line, Signal line, MACD histogram)
    """
    fast_ema = data.ewm(span=fast_period, adjust=False).mean()
    slow_ema = data.ewm(span=slow_period, adjust=False).mean()
    
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

def calculate_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average of volume
    
    Args:
        volume: Volume series data
        period: SMA period
    
    Returns:
        Volume SMA as pandas Series
    """
    return volume.rolling(window=period).mean()

def get_trading_signals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Generate trading signals based on RSI, MACD, and volume indicators
    
    Args:
        df: DataFrame with OHLCV data
        config: Configuration dictionary with indicator parameters
    
    Returns:
        DataFrame with added signal columns
    """
    # Calculate RSI
    df['rsi'] = calculate_rsi(df['close'], config['RSI_PERIOD'])
    
    # Calculate MACD
    macd_line, signal_line, macd_hist = calculate_macd(
        df['close'],
        config['MACD_FAST'],
        config['MACD_SLOW'],
        config['MACD_SIGNAL']
    )
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_hist
    
    # Calculate Volume SMA
    df['volume_sma'] = calculate_volume_sma(df['volume'], config['VOLUME_PERIOD'])
    
    # Generate trading signals
    df['buy_signal'] = (
        (df['rsi'] < config['RSI_OVERSOLD']) &  # RSI oversold
        (df['macd_hist'] > 0) &                 # MACD histogram positive
        (df['volume'] > df['volume_sma'])       # Volume confirmation
    )
    
    df['sell_signal'] = (
        (df['rsi'] > config['RSI_OVERBOUGHT']) &  # RSI overbought
        (df['macd_hist'] < 0) &                   # MACD histogram negative
        (df['volume'] > df['volume_sma'])         # Volume confirmation
    )
    
    return df
