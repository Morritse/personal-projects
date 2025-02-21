import backtrader as bt
import pandas as pd
import numpy as np
from scipy import stats
import pytz

class EnsembleStrategy(bt.Strategy):

    # --------------------------
    # 1) Short-Term RSI/Stoch Params
    # --------------------------
    params = dict(
        # RSI
        rsi_period=2,
        rsi_overbought=90,
        rsi_oversold=10,

        # Stoch
        stoch_period=5,
        stoch_period_d=2,
        stoch_period_k=2,
        stoch_overbought=90,
        stoch_oversold=10,

        # 1-minute forward periods
        short_forward_periods=[1, 3, 10],

        # --------------------------
        # 2) Medium-Term MACD/EMA Params
        # --------------------------
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,

        ema_period=20,

        # 5-minute forward periods
        medium_forward_periods=[5, 15, 30],

        macd_std_threshold=1.5,
        min_trend_strength=0.003,
        momentum_period=5,

        volume_ma_period=20,
        min_volume_mult=1.2,  # reduce/increase as you want

        # --------------------------
        # Market Hours
        # (Used by both short + medium logic)
        # Skip first/last 15 min if you want
        # --------------------------
        market_open_hour=9,
        market_open_min=30,
        market_close_hour=16,
        market_close_min=0,
        skip_open_minutes=15,
        skip_close_minutes=15,
    )

    def __init__(self):
        """
        We expect:
          - self.datas[0]: 1-minute data
          - self.datas[1]: 5-minute resampled data
        """
        # -----------------------------------------------------
        # SHORT-TERM (1-min) INDICATORS
        # -----------------------------------------------------
        self.data1m = self.datas[0]
        # RSI
        self.rsi = bt.indicators.RSI(
            self.data1m,
            period=self.p.rsi_period,
            safediv=True
        )
        # Stochastic
        self.stoch = bt.indicators.Stochastic(
            self.data1m,
            period=self.p.stoch_period,
            period_dfast=self.p.stoch_period_d,
            period_dslow=self.p.stoch_period_k,
            safediv=True
        )

        # We'll store short-term signals here
        self.short_signals = []

        # -----------------------------------------------------
        # MEDIUM-TERM (5-min) INDICATORS
        # -----------------------------------------------------
        self.data5m = self.datas[1]

        self.macd = bt.indicators.MACD(
            self.data5m,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
            plot=False
        )
        self.ema = bt.indicators.ExponentialMovingAverage(
            self.data5m, 
            period=self.p.ema_period, 
            plot=False
        )
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data5m.volume,
            period=self.p.volume_ma_period
        )

        # For tracking MACD threshold
        self.macd_diffs = []

        # We'll store medium-term signals here
        self.medium_signals = []

        # -----------------------------------------------------
        # MISC TRACKING
        # -----------------------------------------------------
        self.ny_tz = pytz.timezone("US/Eastern")
        self.last_bar_time_1m = None
        self.last_bar_time_5m = None

    def is_market_hours(self, dt_utc):
        """
        Convert dt_utc (naive UTC from Backtrader) to Eastern Time
        and check if within market hours (9:30–16:00 ET).
        Optionally skip first/last 'N' minutes after open/before close.
        """
        # Make dt_utc timezone-aware
        dt_utc_aware = dt_utc.replace(tzinfo=pytz.UTC)
        # Convert to Eastern Time
        dt_et = dt_utc_aware.astimezone(self.ny_tz)

        hour = dt_et.hour
        minute = dt_et.minute

        # Basic bounding for 9:30 to 16:00
        if hour < self.p.market_open_hour or hour > self.p.market_close_hour:
            return False
        if hour == self.p.market_open_hour and minute < self.p.market_open_min:
            return False
        if hour == self.p.market_close_hour and minute >= self.p.market_close_min:
            return False

        # Skip first X minutes after open
        if (hour == self.p.market_open_hour and
            minute < (self.p.market_open_min + self.p.skip_open_minutes)):
            return False

        # Skip last X minutes before close
        if (hour == self.p.market_close_hour and
            minute >= (self.p.market_close_min - self.p.skip_close_minutes)):
            return False

        return True

    def next(self):
        """We handle signals from BOTH data0 (1-min) and data1 (5-min)."""
        # 1) If we are on the 1-min data ...
        if self.data0._name == self.data1m._name and len(self.data0) > 0:
            self.handle_short_term()

        # 2) If we are on the 5-min data ...
        if self.data1._name == self.data5m._name and len(self.data1) > 0:
            self.handle_medium_term()

    def handle_short_term(self):
        """Implements short-term (RSI/Stoch) logic on 1-minute bars."""
        dt_utc = self.data1m.datetime.datetime(0)
        if not self.is_market_hours(dt_utc):
            return

        # Build RSI signal
        if self.rsi[0] > self.p.rsi_overbought:
            rsi_sig = -1  # Overbought
        elif self.rsi[0] < self.p.rsi_oversold:
            rsi_sig = 1   # Oversold
        else:
            rsi_sig = 0

        # Build Stoch signal
        if self.stoch.percK[0] > self.p.stoch_overbought:
            stoch_sig = -1
        elif self.stoch.percK[0] < self.p.stoch_oversold:
            stoch_sig = 1
        else:
            stoch_sig = 0

        # Combine signals if you want them to "agree"
        combined_sig = 0
        if rsi_sig == stoch_sig and rsi_sig != 0:
            combined_sig = rsi_sig

        # Store
        self.short_signals.append({
            'bar_index': len(self.data1m) - 1,
            'datetime': dt_utc,
            'signal': combined_sig,
            'close': self.data1m.close[0],
        })

    def handle_medium_term(self):
        """Implements medium-term (MACD/EMA) logic on 5-minute bars."""
        dt_utc = self.data5m.datetime.datetime(0)
        if not self.is_market_hours(dt_utc):
            return

        if not self.macd.macd[0] or not self.macd.signal[0] or not self.ema[0]:
            return

        # MACD diff
        macd_diff = self.macd.macd[0] - self.macd.signal[0]
        self.macd_diffs.append(macd_diff)
        if len(self.macd_diffs) < 20:
            return

        vol_ratio = None
        if len(self.volume_ma) > 0 and self.volume_ma[0] != 0:
            vol_ratio = self.data5m.volume[0] / self.volume_ma[0]
            if vol_ratio < self.p.min_volume_mult:
                return

        # dynamic threshold
        macd_threshold = np.std(self.macd_diffs[-20:]) * self.p.macd_std_threshold
        if abs(macd_diff) <= macd_threshold:
            return

        # MACD direction
        macd_signal = 1 if macd_diff > 0 else -1

        # Check EMA direction
        price_to_ema = (self.data5m.close[0] - self.ema[0]) / self.ema[0]
        if abs(price_to_ema) < self.p.min_trend_strength:
            return
        ema_signal = 1 if price_to_ema > 0 else -1

        # Optional momentum check
        if self.p.momentum_period > 0 and len(self.data5m) > self.p.momentum_period:
            momentum = (self.data5m.close[0] - self.data5m.close[-self.p.momentum_period]) / self.data5m.close[-self.p.momentum_period]
            # momentum in same direction
            if (momentum > 0) != (macd_diff > 0):
                return

        if macd_signal == ema_signal:
            # store
            self.medium_signals.append({
                'bar_index': len(self.data5m) - 1,
                'datetime': dt_utc,
                'combined_signal': macd_signal,
                'close': self.data5m.close[0],
                'macd_diff': macd_diff,
                'macd_threshold': macd_threshold,
                'price_to_ema': price_to_ema,
                'volume_ratio': vol_ratio,
            })

    def stop(self):
        """Compute forward returns for each signal set (short vs. medium)."""
        # -----------------------------------
        # 1) SHORT-TERM (1-minute) Analysis
        # -----------------------------------
        if not self.short_signals:
            print("No short-term signals generated!")
        else:
            df_short = pd.DataFrame(self.short_signals)
            close_array_1m = np.array(self.data1m.close.array)

            print("\n=== Short-Term RSI/Stoch Analysis (1m Data) ===")
            print(f"Total 1m Bars: {len(self.data1m)}")
            print(f"Signals Generated: {len(df_short)}")
            oversold_ct = len(df_short[df_short['signal'] == 1])
            overbought_ct = len(df_short[df_short['signal'] == -1])
            print(f"Oversold: {oversold_ct}, Overbought: {overbought_ct}")

            # Evaluate each forward period
            for period in self.p.short_forward_periods:
                col = f'fwd_ret_{period}'
                rets = []
                for idx in df_short['bar_index']:
                    if idx + period < len(close_array_1m):
                        ret = (close_array_1m[idx + period] - close_array_1m[idx]) / close_array_1m[idx]
                    else:
                        ret = np.nan
                    rets.append(ret)
                df_short[col] = rets

                valid = df_short.dropna(subset=[col])
                valid = valid[valid['signal'] != 0]
                if len(valid) < 2:
                    print(f"\nNot enough valid signals for {period}-bar forward returns.")
                    continue

                ic = valid['signal'].corr(valid[col])
                n = len(valid)
                if abs(ic) < 1.0:
                    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
                else:
                    t_stat = np.inf
                    p_val = 0.0

                print(f"\n--- {period}-Bar Forward Returns ---")
                print(f"Number of Valid Signals: {n}")
                print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")
                avg_os = valid[valid['signal'] == 1][col].mean() * 100
                avg_ob = valid[valid['signal'] == -1][col].mean() * 100
                print(f"Avg Return after Oversold: {avg_os:.3f}%")
                print(f"Avg Return after Overbought: {avg_ob:.3f}%")

        # -----------------------------------
        # 2) MEDIUM-TERM (5-minute) Analysis
        # -----------------------------------
        if not self.medium_signals:
            print("No medium-term signals generated!")
            return
        df_med = pd.DataFrame(self.medium_signals)
        close_array_5m = np.array(self.data5m.close.array)

        print("\n=== Medium-Term MACD/EMA Analysis (5m Data) ===")
        print(f"Total 5m Bars: {len(self.data5m)}")
        print(f"Signals Generated: {len(df_med)}")

        bullish_ct = len(df_med[df_med['combined_signal'] == 1])
        bearish_ct = len(df_med[df_med['combined_signal'] == -1])
        print(f"Bullish: {bullish_ct}, Bearish: {bearish_ct}")

        for period in self.p.medium_forward_periods:
            col = f'fwd_ret_{period}'
            rets = []
            for idx in df_med['bar_index']:
                if idx + period < len(close_array_5m):
                    ret = (close_array_5m[idx + period] - close_array_5m[idx]) / close_array_5m[idx]
                else:
                    ret = np.nan
                rets.append(ret)
            df_med[col] = rets

            valid = df_med.dropna(subset=[col])
            if len(valid) < 2:
                print(f"\nNot enough valid signals for {period}-bar forward returns.")
                continue

            ic = valid['combined_signal'].corr(valid[col])
            n = len(valid)
            if abs(ic) < 1.0:
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
            else:
                t_stat = np.inf
                p_val = 0.0

            print(f"\n--- {period}-Bar Forward Returns ---")
            print(f"Number of Valid Signals: {n}")
            print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")
            avg_bull = valid[valid['combined_signal'] == 1][col].mean() * 100
            avg_bear = valid[valid['combined_signal'] == -1][col].mean() * 100
            print(f"Avg Return (Bullish): {avg_bull:.3f}% | (Bearish): {avg_bear:.3f}%")
