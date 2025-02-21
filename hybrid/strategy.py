# strategy.py
import talib
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
import time
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np

class IchimokuSuperTrendMACDStrategy:
    def __init__(self, base_url='https://paper-api.alpaca.markets'):
        load_dotenv('.env')
        
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_API_SECRET')
        
        # If you plan only to do local backtesting, you can skip
        # the credential check or fallback approach entirely
        if not api_key or not api_secret:
            print("Warning: No Alpaca credentials found. We'll only use local data.")
            self.api = None
        else:
            self.api = tradeapi.REST(api_key, api_secret, base_url)
        
        # Create position placeholder if you want, or skip
        self.position = None

    def get_historical_data(self, symbol, timeframe='15Min', lookback=96, use_local=True):
        """
        Attempt to load data from local CSV first (via DataManager).
        If none found and self.api is not None, fallback to Alpaca.
        """
        if use_local:
            from data_manager import DataManager
            dm = DataManager(data_folder='historical_data')  # or your path
            data = dm.load_historical_data(symbol)
            if data is not None and not data.empty:
                return data  # all data from CSV
        
        # Fallback to Alpaca if we didn't find local or it's empty
        if self.api is not None:
            print(f"Falling back to Alpaca for {symbol} data.")
            end = datetime.now(timezone.utc)
            if timeframe.endswith('Min'):
                minutes = int(timeframe[:-3])
                start = end - timedelta(minutes=lookback*minutes)
            elif timeframe.endswith('Hour'):
                hours = int(timeframe[:-4])
                start = end - timedelta(hours=lookback*hours)
            else:
                start = end - timedelta(days=lookback)
            
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start.isoformat(),
                end=end.isoformat()
            ).df
            # Bars from Alpaca often come multi-indexed with (symbol, date).
            # If so, you can fix that with: bars = bars.xs(symbol, level=0)
            if 'symbol' in bars.columns:
                bars.drop(columns='symbol', inplace=True, errors='ignore')
            
            return bars
        
        # if we have no data from local + no API, return None
        return None

    def calculate_ichimoku(self, data):
        if len(data) < 52:
            return {
                'tenkan_sen': pd.Series([np.nan]*len(data), index=data.index),
                'kijun_sen': pd.Series([np.nan]*len(data), index=data.index),
                'cloud': pd.DataFrame({
                    'senkou_span_a': [np.nan]*len(data),
                    'senkou_span_b': [np.nan]*len(data)
                }, index=data.index)
            }
        
        high = data['high']
        low = data['low']
        
        tenkan_sen = (high.rolling(9).max() + low.rolling(9).min()) / 2
        kijun_sen  = (high.rolling(26).max() + low.rolling(26).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
        
        cloud_df = pd.DataFrame({
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b
        }, index=data.index)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'cloud': cloud_df
        }

    def calculate_supertrend(self, data, atr_period=10, multiplier=3):
        high = data['high']
        low  = data['low']
        close= data['close']
        
        atr = talib.ATR(high, low, close, timeperiod=atr_period)
        hl2 = (high + low)/2
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        st = [0]*len(close)
        direction = [1]*len(close)
        
        for i in range(1, len(close)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                direction[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                st[i] = lower_band.iloc[i]
            else:
                st[i] = upper_band.iloc[i]
        
        return st, direction

    def calculate_macd(self, data):
        close = data['close']
        if len(close) < 26:
            n = len(close)
            return (pd.Series([np.nan]*n, index=close.index),
                    pd.Series([np.nan]*n, index=close.index),
                    pd.Series([np.nan]*n, index=close.index))
        
        macd, signal, hist = talib.MACD(close, 12, 26, 9)
        macd.fillna(0, inplace=True)
        signal.fillna(0, inplace=True)
        hist.fillna(0, inplace=True)
        return macd, signal, hist

    def get_position_qty(self, symbol):
        """
        If you're only doing local backtests, this won't matter.
        For real trading, it calls Alpaca to see how many shares are held.
        """
        if self.api:
            try:
                pos = self.api.get_position(symbol)
                return int(pos.qty)
            except:
                return 0
        return 0

    # The rest (run_strategy, execute_trades) might be for *live trading*, not for backtesting.
    # For offline backtesting, see a separate backtester class.
