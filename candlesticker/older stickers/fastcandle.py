"""
Intraday Candlestick + Momentum + Trend Confirmation + Granular Signal
Using Alpaca Paper Account & TA-Lib
"""

import os
import pandas as pd
import numpy as np
import talib
from talib.abstract import Function
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from typing import Dict, List

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_PAPER_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_PAPER_SECRET")
# Paper URL for trading. Alpaca will automatically route data requests to data.alpaca.markets
BASE_URL = "https://paper-api.alpaca.markets"

# Symbols to analyze
SYMBOLS = [
    "SPY",   # S&P 500 ETF
    "QQQ",   # Nasdaq 100 ETF
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "TSLA",  # Tesla
    "AMZN",  # Amazon
    "META",  # Meta
    "GOOGL", # Google
    "NVDA",  # Nvidia
    "XLF"    # Financial Select Sector SPDR
]

# Resample timeframe (5min or 15min). Pandas notation: '5T' = 5 minutes, '15T' = 15 minutes.
RESAMPLE_TIMEFRAME = '5T'  

# How far back to fetch 1-minute bars?
DAYS_LOOKBACK = 2  # For example, 2 days of 1-min data

# List of candlestick patterns from TA-Lib
CDL_PATTERNS = [
    'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
    'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLADVANCEBLOCK',
    'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL',
    'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 'CDLDOJISTAR',
    'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR',
    'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN',
    'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 'CDLHIKKAKE', 'CDLHIKKAKEMOD',
    'CDLHOMINGPIGEON', 'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER',
    'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 'CDLLONGLEGGEDDOJI',
    'CDLLONGLINE', 'CDLMARUBOZU', 'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR',
    'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS',
    'CDLSEPARATINGLINES', 'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP',
    'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 'CDLTASUKIGAP',
    'CDLTHRUSTING', 'CDLTRISTAR', 'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS',
    'CDLXSIDEGAP3METHODS'
]

class IntradayAnalyzer:
    """Fetch 1-min data, resample to higher timeframe, run candlestick + momentum + trend checks."""
    
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=BASE_URL,
            api_version='v2'
        )

    def fetch_1min_data(self, symbol: str, days: int = 2) -> pd.DataFrame:
        """
        Fetch ~'days' of 1-minute bars from Alpaca. 
        If 'days' is large, you might hit data availability limits on free plan.
        """
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=days)

        # Convert to ISO-8601 (RFC-3339) strings
        start_str = start_dt.isoformat() + "Z"
        end_str = end_dt.isoformat() + "Z"

        print(f"Fetching 1-min data for {symbol} from {start_str} to {end_str}...")
        try:
            bars = self.api.get_bars(
                symbol=symbol,
                timeframe=tradeapi.TimeFrame.Minute,  # 1-min bars
                start=start_str,
                end=end_str,
            ).df

            # If multi-index, flatten
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index(level=[0, 1])

            # Rename columns if needed
            rename_map = {}
            for c in bars.columns:
                if c.lower() in ["open", "high", "low", "close", "volume"]:
                    rename_map[c] = c.title()
            bars.rename(columns=rename_map, inplace=True)

            bars.dropna(inplace=True)
            bars.sort_index(inplace=True)
            return bars
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def resample_bars(self, df: pd.DataFrame, rule: str = '5T') -> pd.DataFrame:
        """
        Resample 1-min bars to a higher timeframe, e.g. 5T, 15T, 1H, etc.
        """
        if df.empty:
            return df

        # We want O/H/L/C from first/max/min/last, sum of volume
        df_resampled = df.resample(rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        return df_resampled

    def compute_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute specified candlestick patterns from TA-Lib and produce a combined score.
        """
        if df.empty:
            return df

        # Prepare inputs for TA-Lib
        inputs = {
            'open': df['Open'],
            'high': df['High'],
            'low': df['Low'],
            'close': df['Close']
        }

        for pattern in CDL_PATTERNS:
            try:
                func = Function(pattern)
                df[pattern] = func(inputs)
            except Exception as e:
                print(f"Warning: {pattern} failed => {str(e)}")
                df[pattern] = 0

        # Sum all candlestick outputs into a single column
        df['cdl_sum'] = df[CDL_PATTERNS].sum(axis=1)

        # Convert that sum into a naive -1/0/+1
        # e.g. if the sum > 0 => +1 (bullish), sum < 0 => -1, else 0
        def interpret_cdl(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            return 0

        df['cdl_signal'] = df['cdl_sum'].apply(interpret_cdl)

        return df

    def compute_momentum(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Compute RSI as a momentum indicator. We'll add a simple check: RSI>50 => bullish bias.
        """
        if df.empty:
            return df

        close = df['Close'].values
        df['RSI'] = talib.RSI(close, timeperiod=period)
        return df

    def compute_trend(self, df: pd.DataFrame, short_window=10, long_window=30) -> pd.DataFrame:
        """
        Compute short and long simple moving averages for a trend filter.
        If short_ma > long_ma => bullish trend.
        """
        if df.empty:
            return df

        df['MA_short'] = df['Close'].rolling(short_window).mean()
        df['MA_long'] = df['Close'].rolling(long_window).mean()
        return df

    def combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine candlestick, RSI, and trend signals into a final composite signal
        ranging from -3 to +3, for example. Let's define:
        
        - cdl_signal in [-1, 0, +1]
        - momentum_signal in [-1, 0, +1] (based on RSI)
        - trend_signal in [-1, 0, +1] (based on MA crossover)

        final_signal = cdl_signal + momentum_signal + trend_signal
        """
        if df.empty:
            return df

        # Momentum signal:
        # RSI > 60 => +1, RSI < 40 => -1, else 0
        def interpret_rsi(val):
            if pd.isna(val):
                return 0
            if val > 60:
                return 1
            elif val < 40:
                return -1
            return 0
        
        df['momentum_sig'] = df['RSI'].apply(interpret_rsi)

        # Trend signal:
        # if MA_short > MA_long => +1, else if < => -1, else 0
        def interpret_trend(row):
            if pd.isna(row['MA_short']) or pd.isna(row['MA_long']):
                return 0
            if row['MA_short'] > row['MA_long']:
                return 1
            elif row['MA_short'] < row['MA_long']:
                return -1
            return 0
        
        df['trend_sig'] = df.apply(interpret_trend, axis=1)

        # Final composite
        # cdl_signal: -1..+1
        # momentum_sig: -1..+1
        # trend_sig: -1..+1
        df['final_signal'] = df['cdl_signal'] + df['momentum_sig'] + df['trend_sig']

        return df

    def run_analysis(self, symbol: str):
        """
        End-to-end analysis for 1 symbol:
        1. Fetch 1-min
        2. Resample (5/15-min)
        3. Compute candlesticks, RSI, MAs
        4. Combine signals -> final_signal
        5. Print tail
        """
        # 1. Fetch
        df_1m = self.fetch_1min_data(symbol, days=DAYS_LOOKBACK)
        if df_1m.empty:
            print(f"No 1-min data for {symbol}, skipping.")
            return

        # 2. Resample
        df_res = self.resample_bars(df_1m, rule=RESAMPLE_TIMEFRAME)
        if df_res.empty:
            print(f"Resampling returned empty for {symbol}, skipping.")
            return

        # 3. Candlesticks
        df_res = self.compute_candlestick_patterns(df_res)
        # 4. Momentum
        df_res = self.compute_momentum(df_res, period=14)
        # 5. Trend
        df_res = self.compute_trend(df_res, short_window=10, long_window=30)
        # 6. Combine signals
        df_res = self.combine_signals(df_res)

        # Show the last few rows
        show_cols = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'cdl_sum', 'cdl_signal', 'RSI', 'MA_short', 'MA_long',
            'momentum_sig', 'trend_sig', 'final_signal'
        ]
        existing = [c for c in show_cols if c in df_res.columns]
        print(f"\n---- {symbol} (last 10 bars) ----\n")
        print(df_res[existing].tail(10))

        # Distribution of final signals
        print("\nFinal Signal distribution:")
        print(df_res['final_signal'].value_counts(dropna=False))

def main():
    analyzer = IntradayAnalyzer()

    for sym in SYMBOLS:
        print("====================================================")
        print(f"Running analysis for {sym}")
        print("====================================================")
        analyzer.run_analysis(sym)

if __name__ == "__main__":
    main()
