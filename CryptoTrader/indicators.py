# indicators.py

import pandas as pd
import talib
from talib import abstract
import numpy as np

def compute_indicators_and_signal(
    df,
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_period=20,
    bb_devup=2.0,
    bb_devdn=2.0
):
    """
    Given a DataFrame with columns: [open, high, low, close, volume],
    compute TA-Lib indicators (RSI, MACD, Bollinger Bands) using 
    specified periods and produce a single 'combined_signal' in [-1, +1].
    """
    from talib import abstract
    import numpy as np

    inputs = {
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume']
    }

    # RSI
    df['rsi'] = abstract.Function('RSI')(inputs, timeperiod=rsi_period)

    # MACD
    macd_func = abstract.Function('MACD')
    macd_out = macd_func(
        inputs,
        fastperiod=macd_fast,
        slowperiod=macd_slow,
        signalperiod=macd_signal
    )
    df['macd'], df['macd_signal'], df['macd_hist'] = macd_out

    # Bollinger Bands
    bb_func = abstract.Function('BBANDS')
    bb_out = bb_func(
        inputs,
        timeperiod=bb_period,
        nbdevup=bb_devup,
        nbdevdn=bb_devdn,
        matype=0  # 0 => simple moving average
    )
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = bb_out

    df.dropna(inplace=True)

    # Normalize RSI: [0..100] -> [-1..+1]
    df['rsi_norm'] = 2 * ((df['rsi'] - 50) / 100)

    # MACD histogram => tanh
    scaling_factor = df['macd_hist'].std() or 1e-9
    df['macd_hist_norm'] = np.tanh(df['macd_hist'] / scaling_factor)

    # Bollinger %b => [-1..+1]
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_norm'] = 2.0 * (df['bb_percent_b'] - 0.5)

    # Combine equally
    df['raw_signal'] = df['rsi_norm'] + df['macd_hist_norm'] + df['bb_norm']
    df['combined_signal'] = df['raw_signal'] / 3.0

    return df
