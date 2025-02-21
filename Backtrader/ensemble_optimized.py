import backtrader as bt
import pandas as pd
import numpy as np
from scipy import stats
import pytz

class EnsembleStrategy(bt.Strategy):
    """
    Optimized ensemble strategy combining:
    1. Short-term (1-min) RSI/Stoch signals
    2. Medium-term (5-min) MACD/EMA signals
    3. Long-term (15-min) SMA/ADX signals
    """
    params = dict(
        # --------------------------
        # Short-Term RSI/Stoch
        # --------------------------
        rsi_period=2,
        rsi_overbought=90,
        rsi_oversold=10,
        stoch_period=5,
        stoch_period_d=2,
        stoch_period_k=2,
        stoch_overbought=90,
        stoch_oversold=10,
        short_forward_periods=[1, 3, 10],

        # --------------------------
        # Medium-Term MACD/EMA
        # --------------------------
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        ema_period=20,
        medium_forward_periods=[5, 15, 30],
        macd_std_threshold=1.5,
        min_trend_strength=0.003,
        momentum_period=5,
        volume_ma_period=20,
        min_volume_mult=1.2,

        # --------------------------
        # Long-Term SMA/ADX
        # --------------------------
        sma_period=200,
        adx_period=14,
        adx_low=15,
        adx_high=25,
        long_forward_periods=[15, 30, 60],

        # --------------------------
        # Market Hours (ET)
        # --------------------------
        market_open_hour=9,
        market_open_min=30,
        market_close_hour=16,
        market_close_min=0,
        skip_open_minutes=15,
        skip_close_minutes=15,

        # --------------------------
        # Ensemble weighting
        # --------------------------
        short_weight=0.2,
        medium_weight=0.4,
        long_weight=0.4,
        short_signal_window=15,
    )

    def __init__(self):
        # ---------------------------------------------------------
        # 1) Data feeds (assuming you have data0 = 1m, data1 = 5m, data2 = 15m)
        # ---------------------------------------------------------
        self.data1m = self.datas[0]   # 1-min feed
        self.data5m = self.datas[1]   # 5-min feed
        self.data15m = self.datas[2]  # 15-min feed

        # ---------------------------------------------------------
        # 2) Indicators and storage per timeframe
        # ---------------------------------------------------------
        self.init_short_term()
        self.init_medium_term()
        self.init_long_term()

        # Timezone for market hours check
        self.ny_tz = pytz.timezone("US/Eastern")

        # Track last bar processed for each timeframe
        self.last_bar_1m = -1  # so we know when a new 1-min bar is here
        self.last_bar_5m = -1  # so we know when a new 5-min bar is here
        self.last_bar_15m = -1 # so we know when a new 15-min bar is here

    # --------------------------
    # Initialize 1m short-term
    # --------------------------
    def init_short_term(self):
        self.rsi = bt.indicators.RSI(
            self.data1m,
            period=self.p.rsi_period,
            safediv=True
        )
        self.stoch = bt.indicators.Stochastic(
            self.data1m,
            period=self.p.stoch_period,
            period_dfast=self.p.stoch_period_d,
            period_dslow=self.p.stoch_period_k,
            safediv=True
        )
        self.short_signals = []
        self.short_signal_buffer = []

    # --------------------------
    # Initialize 5m medium-term
    # --------------------------
    def init_medium_term(self):
        self.macd = bt.indicators.MACD(
            self.data5m,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        self.ema = bt.indicators.ExponentialMovingAverage(
            self.data5m,
            period=self.p.ema_period
        )
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data5m.volume,
            period=self.p.volume_ma_period
        )
        self.macd_diffs = []
        self.medium_signals = []

    # --------------------------
    # Initialize 15m long-term
    # --------------------------
    def init_long_term(self):
        self.long_sma = bt.indicators.SimpleMovingAverage(
            self.data15m,
            period=self.p.sma_period
        )
        self.long_adx = bt.indicators.ADX(
            self.data15m,
            period=self.p.adx_period
        )
        self.long_signals = []
        self.ensemble_signals = []

    # --------------------------
    # Utility: Market Hours
    # --------------------------
    def is_market_hours(self, dt_utc):
        dt_utc_aware = dt_utc.replace(tzinfo=pytz.UTC)
        dt_et = dt_utc_aware.astimezone(self.ny_tz)
        hour, minute = dt_et.hour, dt_et.minute

        if hour < self.p.market_open_hour or hour > self.p.market_close_hour:
            return False
        if hour == self.p.market_open_hour and minute < self.p.market_open_min:
            return False
        if hour == self.p.market_close_hour and minute >= self.p.market_close_min:
            return False

        # skip first X minutes
        if (hour == self.p.market_open_hour and
            minute < (self.p.market_open_min + self.p.skip_open_minutes)):
            return False

        # skip last X minutes
        if (hour == self.p.market_close_hour and
            minute >= (self.p.market_close_min - self.p.skip_close_minutes)):
            return False

        return True

    # ---------------------------------------------------------
    # next() logic: We only process each timeframe when a *new* bar
    # is detected for that timeframe. This ensures each set of signals
    # triggers once per bar in that timeframe.
    # ---------------------------------------------------------
    def next(self):
        # 1) Check if we have a new 1-minute bar
        if len(self.data1m) - 1 > self.last_bar_1m:
            self.last_bar_1m = len(self.data1m) - 1
            self.handle_short_term()

        # 2) Check if we have a new 5-minute bar
        if len(self.data5m) - 1 > self.last_bar_5m:
            self.last_bar_5m = len(self.data5m) - 1
            self.handle_medium_term()

        # 3) Check if we have a new 15-minute bar
        if len(self.data15m) - 1 > self.last_bar_15m:
            self.last_bar_15m = len(self.data15m) - 1
            self.handle_long_term()
            self.combine_signals()

    # -------------------------------------------------------------------
    # 1) SHORT-TERM Logic (called once per new 1-min bar)
    # -------------------------------------------------------------------
    def handle_short_term(self):
        dt_utc = self.data1m.datetime.datetime(0)
        if not self.is_market_hours(dt_utc):
            return

        # RSI
        if self.rsi[0] > self.p.rsi_overbought:
            rsi_sig = -1
        elif self.rsi[0] < self.p.rsi_oversold:
            rsi_sig = 1
        else:
            rsi_sig = 0

        # Stoch
        if self.stoch.percK[0] > self.p.stoch_overbought:
            stoch_sig = -1
        elif self.stoch.percK[0] < self.p.stoch_oversold:
            stoch_sig = 1
        else:
            stoch_sig = 0

        # Combine RSI & Stoch if they agree
        combined_sig = 0
        if (rsi_sig == stoch_sig) and (rsi_sig != 0):
            combined_sig = rsi_sig
            # store in rolling buffer for reference in next 15 minutes
            self.short_signal_buffer.append({
                'datetime': dt_utc,
                'signal': combined_sig
            })

        # For analysis
        self.short_signals.append({
            'bar_index': len(self.data1m)-1,
            'datetime': dt_utc,
            'signal': combined_sig,
            'close': self.data1m.close[0],
        })

    # -------------------------------------------------------------------
    # 2) MEDIUM-TERM Logic (called once per new 5-min bar)
    # -------------------------------------------------------------------
    def handle_medium_term(self):
        dt_utc = self.data5m.datetime.datetime(0)
        if not self.is_market_hours(dt_utc):
            return

        # Ensure indicators are valid
        if not (self.macd.macd[0] and self.macd.signal[0] and self.ema[0]):
            return

        # MACD threshold
        macd_diff = self.macd.macd[0] - self.macd.signal[0]
        self.macd_diffs.append(macd_diff)
        if len(self.macd_diffs) < 20:
            return

        # Volume check
        if self.volume_ma[0] and self.volume_ma[0] > 0:
            vol_ratio = self.data5m.volume[0] / self.volume_ma[0]
            if vol_ratio < self.p.min_volume_mult:
                return
        else:
            vol_ratio = None

        threshold = np.std(self.macd_diffs[-20:]) * self.p.macd_std_threshold
        if abs(macd_diff) <= threshold:
            return

        macd_sig = 1 if macd_diff > 0 else -1

        # Compare to EMA
        price_to_ema = 0
        if self.ema[0]:
            price_to_ema = (self.data5m.close[0] - self.ema[0]) / self.ema[0]
        if abs(price_to_ema) < self.p.min_trend_strength:
            return
        ema_sig = 1 if price_to_ema > 0 else -1

        # Optional momentum check
        if self.p.momentum_period > 0 and len(self.data5m) > self.p.momentum_period:
            mom_ret = (self.data5m.close[0] - self.data5m.close[-self.p.momentum_period]) / self.data5m.close[-self.p.momentum_period]
            if (mom_ret > 0) != (macd_diff > 0):
                return

        if macd_sig == ema_sig:
            combined_sig = macd_sig
            self.medium_signals.append({
                'bar_index': len(self.data5m)-1,
                'datetime': dt_utc,
                'combined_signal': combined_sig,
                'close': self.data5m.close[0],
                'macd_diff': macd_diff,
                'macd_threshold': threshold,
                'price_to_ema': price_to_ema,
                'volume_ratio': vol_ratio
            })

    # -------------------------------------------------------------------
    # 3) LONG-TERM Logic (called once per new 15-min bar)
    # -------------------------------------------------------------------
    def handle_long_term(self):
        dt_utc = self.data15m.datetime.datetime(0)
        if not self.is_market_hours(dt_utc):
            return

        # Wait for valid indicators
        if not (self.long_sma[0] and self.long_adx[0]):
            return

        base_signal = 1 if self.data15m.close[0] > self.long_sma[0] else -1

        # ADX-based scaling
        adx_val = self.long_adx[0]
        if adx_val < self.p.adx_low:
            final_signal = 0.5 * base_signal
        elif adx_val > self.p.adx_high:
            final_signal = 1.0 * base_signal
        else:
            final_signal = 0.75 * base_signal

        self.long_signals.append({
            'bar_index': len(self.data15m)-1,
            'datetime': dt_utc,
            'base_signal': base_signal,
            'adx_val': adx_val,
            'final_signal': final_signal,
            'close': self.data15m.close[0],
        })

    # -------------------------------------------------------------------
    # Combine signals once per new 15-min bar
    # -------------------------------------------------------------------
    def combine_signals(self):
        if not self.long_signals:
            return

        long_latest = self.long_signals[-1]
        dt_utc_15m = long_latest['datetime']
        long_val = long_latest['final_signal']

        # Find recent medium signal
        med_val = 0
        # get any medium signals that occurred within this 15-min bar
        recent_meds = [
            s for s in self.medium_signals
            if s['datetime'] == dt_utc_15m
        ]
        # If none found at the exact bar close time,
        # you might pick the last known medium signal
        if not recent_meds and self.medium_signals:
            # fallback to the most recent medium signal before dt_utc_15m
            # but do not exceed 15 minutes prior if you prefer
            last_signal = max(self.medium_signals, key=lambda x: x['datetime'])
            if last_signal['datetime'] <= dt_utc_15m:
                med_val = last_signal['combined_signal']
        elif recent_meds:
            med_val = recent_meds[-1]['combined_signal']

        # Find recent short signal (within short_signal_window minutes)
        short_val = 0
        cutoff_time = dt_utc_15m - pd.Timedelta(minutes=self.p.short_signal_window)
        # Keep only short signals in that window
        self.short_signal_buffer = [
            s for s in self.short_signal_buffer
            if s['datetime'] >= cutoff_time
        ]
        # latest short signal at or before dt_utc_15m
        valid_shorts = [
            s for s in self.short_signal_buffer
            if s['datetime'] <= dt_utc_15m
        ]
        if valid_shorts:
            # pick the newest
            valid_shorts.sort(key=lambda x: x['datetime'], reverse=True)
            short_val = valid_shorts[0]['signal']

        # Weighted ensemble
        ensemble_val = (
            self.p.short_weight * short_val +
            self.p.medium_weight * med_val +
            self.p.long_weight * long_val
        )

        # Store if significant
        if abs(ensemble_val) > 0.01:
            self.ensemble_signals.append({
                'bar_index': long_latest['bar_index'],
                'datetime': dt_utc_15m,
                'short_signal': short_val,
                'medium_signal': med_val,
                'long_signal': long_val,
                'ensemble_signal': ensemble_val,
                'close': long_latest['close'],
            })

    # -------------------------------------------------------------------
    # STOP: Analysis of signals
    # -------------------------------------------------------------------
    def stop(self):
        symbol = self.data1m._name
        print(f"\n=== Analysis Results for {symbol} ===")

        # Print short signal counts
        if self.short_signals:
            oversold = sum(1 for s in self.short_signals if s['signal'] == 1)
            overbought = sum(1 for s in self.short_signals if s['signal'] == -1)
            print(f"\nShort-term Signals => Oversold: {oversold}, Overbought: {overbought}")

        # Medium
        if self.medium_signals:
            bullish = sum(1 for s in self.medium_signals if s['combined_signal'] == 1)
            bearish = sum(1 for s in self.medium_signals if s['combined_signal'] == -1)
            print(f"Medium-term Signals => Bullish: {bullish}, Bearish: {bearish}")

        # Long
        if self.long_signals:
            bullish = sum(1 for s in self.long_signals if s['final_signal'] > 0)
            bearish = sum(1 for s in self.long_signals if s['final_signal'] < 0)
            print(f"Long-term Signals => Bullish: {bullish}, Bearish: {bearish}")

        # Ensemble
        if not self.ensemble_signals:
            print("No ensemble signals generated!")
            return

        bull_ens = sum(1 for s in self.ensemble_signals if s['ensemble_signal'] > 0)
        bear_ens = sum(1 for s in self.ensemble_signals if s['ensemble_signal'] < 0)
        print(f"Ensemble => Bullish: {bull_ens}, Bearish: {bear_ens}")

        # Evaluate forward returns on the 15-min data
        df_ens = pd.DataFrame(self.ensemble_signals)
        close_arr_15m = np.array(self.data15m.close.array)

        for period in self.p.long_forward_periods:
            col = f'fwd_ret_{period}'
            rets = []
            for idx in df_ens['bar_index']:
                if idx + period < len(close_arr_15m):
                    ret = (close_arr_15m[idx + period] - close_arr_15m[idx]) / close_arr_15m[idx]
                else:
                    ret = np.nan
                rets.append(ret)
            df_ens[col] = rets

            valid = df_ens.dropna(subset=[col])
            valid = valid[abs(valid['ensemble_signal']) > 0.01]

            if len(valid) < 2:
                print(f"\nNot enough valid ensemble signals for {period}-bar forward returns.")
                continue

            ic = valid['ensemble_signal'].corr(valid[col])
            avg_pos = valid[valid['ensemble_signal'] > 0][col].mean() * 100
            avg_neg = valid[valid['ensemble_signal'] < 0][col].mean() * 100

            print(f"\n--- {period}-Bar Forward Returns ---")
            print(f"Number of Valid Signals: {len(valid)}")
            print(f"Information Coefficient (IC): {ic:.3f}")
            print(f"Avg Return (Bullish): {avg_pos:.3f}% | (Bearish): {avg_neg:.3f}%")
