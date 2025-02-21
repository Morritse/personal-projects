"""
Intraday Candlestick + Momentum + Trend Confirmation + Granular Signal
Using Alpaca Paper Account & TA-Lib for Crypto Symbols
But returning only the *latest* final_signal (no naive historical trades).
"""

import os
import pandas as pd
import numpy as np
import talib
from talib.abstract import Function
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
from typing import List, Dict

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_PAPER_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_PAPER_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

# Default: 1-minute bars for crypto
DAYS_LOOKBACK = 2  # e.g., 2 days of 1-min data
RESAMPLE_TIMEFRAME = '5min'  # resample to 5-min bars
CRYPTO_EXCHANGE = "CBSE"   # or whatever exchange you prefer

# Candlestick patterns from TA-Lib
CDL_PATTERNS = [
    'CDL2CROWS','CDL3BLACKCROWS','CDL3INSIDE','CDL3LINESTRIKE','CDL3OUTSIDE',
    'CDL3STARSINSOUTH','CDL3WHITESOLDIERS','CDLADVANCEBLOCK','CDLBELTHOLD',
    'CDLBREAKAWAY','CDLCLOSINGMARUBOZU','CDLCONCEALBABYSWALL','CDLCOUNTERATTACK',
    'CDLDARKCLOUDCOVER','CDLDOJI','CDLDOJISTAR','CDLDRAGONFLYDOJI','CDLENGULFING',
    'CDLEVENINGDOJISTAR','CDLEVENINGSTAR','CDLGAPSIDESIDEWHITE','CDLGRAVESTONEDOJI',
    'CDLHAMMER','CDLHANGINGMAN','CDLHARAMI','CDLHARAMICROSS','CDLHIGHWAVE',
    'CDLHIKKAKE','CDLHIKKAKEMOD','CDLHOMINGPIGEON','CDLIDENTICAL3CROWS','CDLINNECK',
    'CDLINVERTEDHAMMER','CDLKICKING','CDLKICKINGBYLENGTH','CDLLADDERBOTTOM',
    'CDLLONGLEGGEDDOJI','CDLLONGLINE','CDLMARUBOZU','CDLMATCHINGLOW','CDLMATHOLD',
    'CDLMORNINGDOJISTAR','CDLMORNINGSTAR','CDLONNECK','CDLPIERCING','CDLRICKSHAWMAN',
    'CDLRISEFALL3METHODS','CDLSEPARATINGLINES','CDLSHOOTINGSTAR','CDLSHORTLINE',
    'CDLSPINNINGTOP','CDLSTALLEDPATTERN','CDLSTICKSANDWICH','CDLTAKURI','CDLTASUKIGAP',
    'CDLTHRUSTING','CDLTRISTAR','CDLUNIQUE3RIVER','CDLUPSIDEGAP2CROWS','CDLXSIDEGAP3METHODS'
]

class IntradayCryptoAnalyzer:
    """
    Fetch 1-min crypto data, resample, compute candlestick patterns, RSI, MAs, 
    combine into a final_signal, but DO NOT place naive trades on historical bars.
    Instead, we just return the final DataFrame or final_signal for the *latest bar*.
    """

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=BASE_URL,
            api_version='v2'
        )
        # We'll store the final resampled DataFrame for each run in self.df_res
        self.df_res: pd.DataFrame = pd.DataFrame()

    def fetch_1min_data(self, symbol: str, days: int = 2) -> pd.DataFrame:
        """
        Fetch ~'days' of 1-minute bars for crypto from Alpaca using get_crypto_bars.
        Note: symbol is e.g. "BTC/USD" (with slash), but for the API call we typically remove the slash.
        """
        # remove slash for API call, e.g. "BTCUSD"
        symbol_api = symbol

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str   = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        print(f"Fetching 1-min CRYPTO data for {symbol} from {start_str} to {end_str} on {CRYPTO_EXCHANGE} exchange...")

        try:
            bars = self.api.get_crypto_bars(
                symbol=symbol_api,
                timeframe=tradeapi.TimeFrame.Minute,
                start=start_str,
                end=end_str,
                # exchanges=[CRYPTO_EXCHANGE],  # optional
            ).df

            if bars.empty:
                print(f"No crypto bars returned for {symbol}.")
                return pd.DataFrame()

            # If multi-index, flatten
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index(level=0, drop=True)

            # Ensure index is tz-aware
            if not bars.index.tz:
                bars.index = bars.index.tz_localize('UTC')

            # Standardize column names
            bars.columns = [c.capitalize() for c in bars.columns]
            bars.dropna(inplace=True)
            bars.sort_index(inplace=True)
            return bars
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def resample_bars(self, df: pd.DataFrame, rule: str = '5min') -> pd.DataFrame:
        """
        Resample 1-min crypto bars to a higher timeframe.
        """
        if df.empty:
            return df

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
        Compute candlestick patterns from TA-Lib and produce a combined score.
        """
        if df.empty:
            return df

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
                # If some pattern fails, fill with 0
                df[pattern] = 0

        # sum them
        df['cdl_sum'] = df[CDL_PATTERNS].sum(axis=1)

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
        Compute RSI for momentum.
        """
        if df.empty:
            return df

        close = df['Close'].values
        df['RSI'] = talib.RSI(close, timeperiod=period)
        return df

    def compute_trend(self, df: pd.DataFrame, short_window=10, long_window=30) -> pd.DataFrame:
        """
        Compute short and long MAs for a trend filter.
        """
        if df.empty:
            return df

        df['MA_short'] = df['Close'].rolling(short_window).mean()
        df['MA_long']  = df['Close'].rolling(long_window).mean()
        return df

    def combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Combine cdl_signal, RSI, and trend signals into final_signal in range [-3..+3].
        """
        if df.empty:
            return df

        def interpret_rsi(val):
            if pd.isna(val):
                return 0
            if val > 60:
                return 1
            elif val < 40:
                return -1
            return 0
        
        df['momentum_sig'] = df['RSI'].apply(interpret_rsi)

        def interpret_trend(row):
            if pd.isna(row['MA_short']) or pd.isna(row['MA_long']):
                return 0
            if row['MA_short'] > row['MA_long']:
                return 1
            elif row['MA_short'] < row['MA_long']:
                return -1
            return 0
        
        df['trend_sig'] = df.apply(interpret_trend, axis=1)
        # final_signal = cdl_signal + momentum_sig + trend_sig
        df['final_signal'] = df['cdl_signal'] + df['momentum_sig'] + df['trend_sig']
        return df

    # ---------------------------------------------------------
    # MAIN ENTRYPOINT: compute the final Signal for the latest bar
    # ---------------------------------------------------------
    def get_latest_signal(self, symbol: str) -> float:
        """
        1) Fetch ~2 days of 1-min bars
        2) Resample to 5-min
        3) Compute candlestick patterns, RSI, MAs, final_signal
        4) Return final_signal of the *latest* bar
        """

        df_1m = self.fetch_1min_data(symbol, days=DAYS_LOOKBACK)
        if df_1m.empty:
            print(f"No data or error for {symbol}, returning final_signal=0.")
            return 0.0

        df_res = self.resample_bars(df_1m, rule=RESAMPLE_TIMEFRAME)
        if df_res.empty:
            print(f"Resample returned empty for {symbol}, final_signal=0.")
            return 0.0

        df_res = self.compute_candlestick_patterns(df_res)
        df_res = self.compute_momentum(df_res, period=14)
        df_res = self.compute_trend(df_res, short_window=10, long_window=30)
        df_res = self.combine_signals(df_res)

        self.df_res = df_res  # store for debugging if needed

        # get the last bar's final_signal
        last_idx = df_res.index[-1]
        final_sig = df_res.loc[last_idx, 'final_signal']
        print(f"{symbol}: final_signal for latest bar => {final_sig}")
        return float(final_sig)
