# cryptosticker.py

import os
import pandas as pd
import numpy as np
import talib
from talib.abstract import Function
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
from typing import List, Dict

# ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_PAPER_KEY")
# ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_PAPER_SECRET")
ALPACA_API_KEY="PK9GGP2I0PAS11A7IOQB"
ALPACA_SECRET_KEY="BcAYtWDkr11tpzCtBqC3Cud2VQ6VuXH8o3BH6Kg0"
BASE_URL = "https://paper-api.alpaca.markets"

# We'll assume you want 2 days of 1-min data, then resample to 5-min
DAYS_LOOKBACK = 2
RESAMPLE_TIMEFRAME = '5min'

# A big list of TA-Lib candlestick patterns
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
    Fetch 1-min crypto data for a slash-based symbol (e.g. 'BTC/USD'), 
    resample to 5-min, compute candlestick patterns, RSI, MAs, combine into final_signal.
    Return final_signal from the *latest* bar only.
    """

    def __init__(self):
        self.api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=BASE_URL,
            api_version='v2'
        )
        self.df_res: pd.DataFrame = pd.DataFrame()  # store final resampled data

    def fetch_1min_data(self, symbol: str, days: int = DAYS_LOOKBACK) -> pd.DataFrame:
        """
        Uses get_crypto_bars with slash-based symbol. We'll pass e.g. "BTC/USD" directly.
        We then rename columns to match your pipeline.
        """
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        start_str = start_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_str   = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        print(f"Fetching 1-min CRYPTO data for {symbol} from {start_str} to {end_str} ...")

        try:
            # We can pass 'symbol' with slash if your Alpaca version supports it
            bars = self.api.get_crypto_bars(
                symbol=symbol,
                timeframe=tradeapi.TimeFrame.Minute,
                start=start_str,
                end=end_str,
                limit=10000  # or whatever
            ).df

            if bars.empty:
                print(f"No crypto bars returned for {symbol}.")
                return pd.DataFrame()

            # Flatten multi-index if needed
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index(level=0, drop=True)

            if not bars.index.tz:
                bars.index = bars.index.tz_localize('UTC')

            # Rename columns to Title case: Open, High, Low, Close, Volume
            # Because your pipeline expects .title() columns
            rename_map = {}
            for c in bars.columns:
                if c.lower() in ['open','high','low','close','volume']:
                    rename_map[c] = c.capitalize()

            bars.rename(columns=rename_map, inplace=True)
            bars.dropna(inplace=True)
            bars.sort_index(inplace=True)
            return bars

        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()

    def resample_bars(self, df: pd.DataFrame, rule: str = RESAMPLE_TIMEFRAME) -> pd.DataFrame:
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
            except:
                df[pattern] = 0

        df['cdl_sum'] = df[CDL_PATTERNS].sum(axis=1)

        def interpret_cdl(x):
            if x > 0: return 1
            elif x < 0: return -1
            return 0

        df['cdl_signal'] = df['cdl_sum'].apply(interpret_cdl)
        return df

    def compute_momentum(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        if df.empty:
            return df

        close_arr = df['Close'].values
        df['RSI'] = talib.RSI(close_arr, timeperiod=period)
        return df

    def compute_trend(self, df: pd.DataFrame, short_window=10, long_window=30) -> pd.DataFrame:
        if df.empty:
            return df

        df['MA_short'] = df['Close'].rolling(short_window).mean()
        df['MA_long']  = df['Close'].rolling(long_window).mean()
        return df

    def combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        def interpret_rsi(val):
            if pd.isna(val): return 0
            if val > 60: return 1
            if val < 40: return -1
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

        # final_signal in [-3..+3]
        df['final_signal'] = df['cdl_signal'] + df['momentum_sig'] + df['trend_sig']
        return df

    def get_latest_signal(self, symbol: str) -> float:
        """
        1) Fetch ~2 days of 1-min bars with slash-based symbol
        2) Resample
        3) Compute patterns, RSI, trend => final_signal
        4) Return the final_signal of the *most recent* bar
        """
        df_1m = self.fetch_1min_data(symbol, DAYS_LOOKBACK)
        if df_1m.empty:
            print(f"No data for {symbol}, returning 0.")
            return 0.0

        df_res = self.resample_bars(df_1m, RESAMPLE_TIMEFRAME)
        if df_res.empty:
            print(f"No resampled data for {symbol}, returning 0.")
            return 0.0

        df_res = self.compute_candlestick_patterns(df_res)
        df_res = self.compute_momentum(df_res, 14)
        df_res = self.compute_trend(df_res, 10, 30)
        df_res = self.combine_signals(df_res)

        # store for debugging
        self.df_res = df_res

        last_idx = df_res.index[-1]
        final_sig = df_res.loc[last_idx, 'final_signal']
        print(f"{symbol}: final_signal => {final_sig}")
        return float(final_sig)
