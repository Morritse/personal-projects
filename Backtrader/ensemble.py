import backtrader as bt
import pandas as pd
import numpy as np
from scipy import stats
import pytz

class EnsembleStrategy(bt.Strategy):
    """
    Ensemble strategy combining:
    1. Short-term (1-min) RSI/Stoch signals
    2. Medium-term (5-min) MACD/EMA signals
    3. Long-term (15-min) SMA/ADX signals
    """
    params = dict(
        # --------------------------
        # 1) Short-Term RSI/Stoch
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
        # 2) Medium-Term MACD/EMA
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
        # 3) Long-Term SMA/ADX
        # --------------------------
        sma_period=200,     # "200-bar" SMA on 15-min
        adx_period=14,
        adx_low=15,         # threshold for range-bound
        adx_high=25,        # threshold for strong trend
        long_forward_periods=[15, 30, 60],  # measure forward returns

        # --------------------------
        # Market Hours
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
        short_weight=0.2,   # 20% weight to short-term
        medium_weight=0.4,  # 40% weight to medium-term
        long_weight=0.4,    # 40% weight to long-term

        # Keep short signals in the buffer for this many minutes
        short_signal_window=15,  
    )

    def __init__(self, progress_bar=None):
        """Initialize strategy with optional progress bar."""
        # Progress tracking
        self.progress_bar = progress_bar
        self.bar_count = 0
        
        # Initialize data feeds
        self.data1m = self.datas[0]   # 1-min
        self.data5m = self.datas[1]   # 5-min
        self.data15m = self.datas[2]  # 15-min
        
        # Initialize indicators and storage
        self.init_short_term()
        self.init_medium_term()
        self.init_long_term()
        
        # Timezone for market hours
        self.ny_tz = pytz.timezone("US/Eastern")
        
        # Track last processed times
        self.last_5m_bar_time = None
        self.last_15m_bar_time = None
        
    def init_short_term(self):
        """Initialize short-term indicators."""
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
        
    def init_medium_term(self):
        """Initialize medium-term indicators."""
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
        
    def init_long_term(self):
        """Initialize long-term indicators."""
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

    def is_market_hours(self, dt_utc):
        """Check if time is during market hours."""
        dt_utc_aware = dt_utc.replace(tzinfo=pytz.UTC)
        dt_et = dt_utc_aware.astimezone(self.ny_tz)
        hour, minute = dt_et.hour, dt_et.minute

        # Basic bounding: 9:30 to 16:00
        if hour < self.p.market_open_hour or hour > self.p.market_close_hour:
            return False
        if hour == self.p.market_open_hour and minute < self.p.market_open_min:
            return False
        if hour == self.p.market_close_hour and minute >= self.p.market_close_min:
            return False

        # Skip first X minutes
        if (hour == self.p.market_open_hour and
            minute < (self.p.market_open_min + self.p.skip_open_minutes)):
            return False

        # Skip last X minutes
        if (hour == self.p.market_close_hour and
            minute >= (self.p.market_close_min - self.p.skip_close_minutes)):
            return False

        return True

    def next(self):
        """Handle signals from all three timeframes."""
        # Update progress on 1-min bars
        if self.data0._name == self.data1m._name:
            if self.progress_bar:
                self.bar_count += 1
                if self.bar_count % 100 == 0:  # Update every 100 bars
                    self.progress_bar.update(100)
            self.handle_short_term()

        # Process other timeframes
        if self.data1._name == self.data5m._name:
            self.handle_medium_term()

        if len(self.datas) > 2 and self.data2._name == self.data15m._name:
            self.handle_long_term()
            self.combine_signals()

    def handle_short_term(self):
        """Generate short-term signal from RSI/Stoch."""
        dt_utc = self.data1m.datetime.datetime(0)
        if not self.is_market_hours(dt_utc):
            return

        # RSI signal
        if self.rsi[0] > self.p.rsi_overbought:
            rsi_sig = -1
        elif self.rsi[0] < self.p.rsi_oversold:
            rsi_sig = 1
        else:
            rsi_sig = 0

        # Stoch signal    
        if self.stoch.percK[0] > self.p.stoch_overbought:
            stoch_sig = -1
        elif self.stoch.percK[0] < self.p.stoch_oversold:
            stoch_sig = 1
        else:
            stoch_sig = 0

        # Combined signal
        combined_sig = 0
        if (rsi_sig == stoch_sig) and (rsi_sig != 0):
            combined_sig = rsi_sig
            # Add to rolling buffer
            self.short_signal_buffer.append({
                'datetime': dt_utc,
                'signal': combined_sig
            })

        # For analysis/tracking only
        self.short_signals.append({
            'bar_index': len(self.data1m)-1,
            'datetime': dt_utc,
            'signal': combined_sig,
            'close': self.data1m.close[0],
        })

    def handle_medium_term(self):
        """Generate medium-term signal from MACD/EMA."""
        dt_utc = self.data5m.datetime.datetime(0)
        if not self.is_market_hours(dt_utc):
            return

        # Wait for valid indicators
        if not self.macd.macd[0] or not self.macd.signal[0] or not self.ema[0]:
            return

        # Track MACD diffs
        macd_diff = self.macd.macd[0] - self.macd.signal[0]
        self.macd_diffs.append(macd_diff)
        if len(self.macd_diffs) < 20:
            return

        # Volume filter
        vol_ratio = None
        if self.volume_ma[0] and self.volume_ma[0] != 0:
            vol_ratio = self.data5m.volume[0] / self.volume_ma[0]
            if vol_ratio < self.p.min_volume_mult:
                return

        # MACD threshold
        threshold = np.std(self.macd_diffs[-20:]) * self.p.macd_std_threshold
        if abs(macd_diff) <= threshold:
            return

        # Direction
        macd_sig = 1 if macd_diff > 0 else -1
        # Compare to EMA
        price_to_ema = (self.data5m.close[0] - self.ema[0]) / self.ema[0]
        if abs(price_to_ema) < self.p.min_trend_strength:
            return
        ema_sig = 1 if price_to_ema > 0 else -1

        # Optional momentum check
        if self.p.momentum_period > 0 and len(self.data5m) > self.p.momentum_period:
            mom_ret = (self.data5m.close[0] - self.data5m.close[-self.p.momentum_period]) / self.data5m.close[-self.p.momentum_period]
            # Must match MACD direction
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

    def handle_long_term(self):
        """Generate long-term signal from SMA and ADX."""
        dt_utc = self.data15m.datetime.datetime(0)
        if not self.is_market_hours(dt_utc):
            return

        # Skip if indicators aren't ready
        if not self.long_sma[0] or not self.long_adx[0]:
            return

        # Base signal from SMA
        base_signal = 1 if self.data15m.close[0] > self.long_sma[0] else -1

        # Scale by ADX
        adx_val = self.long_adx[0]
        if adx_val < self.p.adx_low:
            final_signal = 0.5 * base_signal  # weak trend
        elif adx_val > self.p.adx_high:
            final_signal = 1.0 * base_signal  # strong trend
        else:
            final_signal = 0.75 * base_signal  # moderate trend

        # Store signal
        self.long_signals.append({
            'bar_index': len(self.data15m)-1,
            'datetime': dt_utc,
            'base_signal': base_signal,
            'adx_val': adx_val,
            'final_signal': final_signal,
            'close': self.data15m.close[0],
        })

    def combine_signals(self):
        """Generate ensemble signal on 15-min bars."""
        if not self.long_signals:
            return

        # Get latest long signal
        long_latest = self.long_signals[-1]
        dt_utc_15m = long_latest['datetime']
        long_val = long_latest['final_signal']

        # Find recent medium signal
        med_val = 0
        if self.medium_signals:
            # Get medium signals in last 15 mins
            recent_meds = [
                s for s in self.medium_signals
                if s['datetime'] > (dt_utc_15m - pd.Timedelta(minutes=15))
                and s['datetime'] <= dt_utc_15m
            ]
            if recent_meds:
                med_val = recent_meds[-1]['combined_signal']

        # Find recent short signal
        short_val = 0
        # Clean old signals
        cutoff_time = dt_utc_15m - pd.Timedelta(minutes=self.p.short_signal_window)
        self.short_signal_buffer = [
            s for s in self.short_signal_buffer
            if s['datetime'] >= cutoff_time
        ]
        # Get most recent short signal
        recent_shorts = [
            s for s in self.short_signal_buffer
            if s['datetime'] <= dt_utc_15m
        ]
        if recent_shorts:
            recent_shorts.sort(key=lambda x: x['datetime'], reverse=True)
            short_val = recent_shorts[0]['signal']

        # Combine signals with weights
        ensemble_val = (
            self.p.short_weight * short_val +
            self.p.medium_weight * med_val +
            self.p.long_weight * long_val
        )

        # Store ensemble aligned to 15-min bar
        if abs(ensemble_val) > 0.01:  # Filter tiny signals
            self.ensemble_signals.append({
                'bar_index': long_latest['bar_index'],
                'datetime': dt_utc_15m,
                'short_signal': short_val,
                'medium_signal': med_val,
                'long_signal': long_val,
                'ensemble_signal': ensemble_val,
                'close': long_latest['close'],
            })

    def stop(self):
        """Analyze signals and compute forward returns."""
        # 1) SHORT
        if len(self.short_signals) > 0:
            self.analyze_short_signals()
        else:
            print("No short-term signals generated.")

        # 2) MEDIUM
        if len(self.medium_signals) > 0:
            self.analyze_medium_signals()
        else:
            print("No medium-term signals generated.")

        # 3) LONG
        if len(self.long_signals) > 0:
            self.analyze_long_signals()
        else:
            print("No long-term signals generated.")

        # 4) ENSEMBLE
        if len(self.ensemble_signals) > 0:
            self.analyze_ensemble_signals()
        else:
            print("No ensemble signals generated.")

    def analyze_short_signals(self):
        """Analyze short-term signals."""
        df_short = pd.DataFrame(self.short_signals)
        close_array_1m = np.array(self.data1m.close.array)

        print("\n=== Short-Term RSI/Stoch Analysis (1m Data) ===")
        print(f"Total 1m Bars: {len(self.data1m)}")
        print(f"Signals Generated: {len(df_short)}")

        oversold_ct = len(df_short[df_short['signal'] == 1])
        overbought_ct = len(df_short[df_short['signal'] == -1])
        print(f"Oversold: {oversold_ct}, Overbought: {overbought_ct}")

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
                print(f"\nNot enough valid short signals for {period}-bar forward returns.")
                continue

            ic = valid['signal'].corr(valid[col])
            n = len(valid)
            p_val = 1.0
            if abs(ic) < 1.0:
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

            print(f"\n--- {period}-Bar Forward Returns (Short) ---")
            print(f"Number of Valid Short Signals: {n}")
            print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")

            # By signal type
            avg_over = valid[valid['signal'] == 1][col].mean() * 100
            avg_under = valid[valid['signal'] == -1][col].mean() * 100
            print(f"Avg Return (Oversold=1): {avg_over:.3f}% | (Overbought=-1): {avg_under:.3f}%")

    def analyze_medium_signals(self):
        """Analyze medium-term signals."""
        df_med = pd.DataFrame(self.medium_signals)
        close_array_5m = np.array(self.data5m.close.array)

        print("\n=== Medium-Term MACD/EMA Analysis (5m Data) ===")
        print(f"Total 5m Bars: {len(self.data5m)}")
        print(f"Signals Generated: {len(df_med)}")

        bull_ct = len(df_med[df_med['combined_signal'] == 1])
        bear_ct = len(df_med[df_med['combined_signal'] == -1])
        print(f"Bullish: {bull_ct}, Bearish: {bear_ct}")

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
                print(f"\nNot enough valid medium signals for {period}-bar forward returns.")
                continue

            ic = valid['combined_signal'].corr(valid[col])
            n = len(valid)
            p_val = 1.0
            if abs(ic) < 1.0:
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

            print(f"\n--- {period}-Bar Forward Returns (Medium) ---")
            print(f"Number of Valid Medium Signals: {n}")
            print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")

            # Averages
            avg_bull = valid[valid['combined_signal'] == 1][col].mean() * 100
            avg_bear = valid[valid['combined_signal'] == -1][col].mean() * 100
            print(f"Avg Return (Bullish=1): {avg_bull:.3f}% | (Bearish=-1): {avg_bear:.3f}%")

    def analyze_long_signals(self):
        """Analyze long-term signals."""
        df_long = pd.DataFrame(self.long_signals)
        close_array_15m = np.array(self.data15m.close.array)

        print("\n=== Long-Term SMA/ADX Analysis (15m Data) ===")
        print(f"Total 15m Bars: {len(self.data15m)}")
        print(f"Signals Generated: {len(df_long)}")

        bull_ct = len(df_long[df_long['final_signal'] > 0])
        bear_ct = len(df_long[df_long['final_signal'] < 0])
        print(f"Bullish: {bull_ct}, Bearish: {bear_ct}")

        for period in self.p.long_forward_periods:
            col = f'fwd_ret_{period}'
            rets = []
            for idx in df_long['bar_index']:
                if idx + period < len(close_array_15m):
                    ret = (close_array_15m[idx + period] - close_array_15m[idx]) / close_array_15m[idx]
                else:
                    ret = np.nan
                rets.append(ret)
            df_long[col] = rets

            valid = df_long.dropna(subset=[col])
            if len(valid) < 2:
                print(f"\nNot enough valid long signals for {period}-bar forward returns.")
                continue

            ic = valid['final_signal'].corr(valid[col])
            n = len(valid)
            p_val = 1.0
            if abs(ic) < 1.0:
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

            print(f"\n--- {period}-Bar Forward Returns (Long) ---")
            print(f"Number of Valid Long Signals: {n}")
            print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")

            avg_bull = valid[valid['final_signal'] > 0][col].mean() * 100
            avg_bear = valid[valid['final_signal'] < 0][col].mean() * 100
            print(f"Avg Return (Bullish>0): {avg_bull:.3f}% | (Bearish<0): {avg_bear:.3f}%")

    def analyze_ensemble_signals(self):
        """Analyze ensemble signals."""
        df_ens = pd.DataFrame(self.ensemble_signals)
        close_array_15m = np.array(self.data15m.close.array)

        print("\n=== Ensemble (Short + Medium + Long) Analysis ===")
        print(f"Ensemble Rows: {len(df_ens)}")
        bull_ct = len(df_ens[df_ens['ensemble_signal'] > 0])
        bear_ct = len(df_ens[df_ens['ensemble_signal'] < 0])
        print(f"Bullish: {bull_ct}, Bearish: {bear_ct}")

        for period in self.p.long_forward_periods:
            col = f'fwd_ret_{period}'
            rets = []
            for idx in df_ens['bar_index']:
                if idx + period < len(close_array_15m):
                    ret = (close_array_15m[idx + period] - close_array_15m[idx]) / close_array_15m[idx]
                else:
                    ret = np.nan
                rets.append(ret)

            df_ens[col] = rets
            valid = df_ens.dropna(subset=[col])
            valid = valid[abs(valid['ensemble_signal']) > 0.01]  # Filter tiny signals
            if len(valid) < 2:
                print(f"\nNot enough valid ensemble signals for {period}-bar forward returns.")
                continue

            ic = valid['ensemble_signal'].corr(valid[col])
            n = len(valid)
            p_val = 1.0
            if abs(ic) < 1.0:
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

            print(f"\n--- {period}-Bar Forward Returns (Ensemble) ---")
            print(f"Number of Valid Ensemble Signals: {n}")
            print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")

            avg_pos = valid[valid['ensemble_signal'] > 0][col].mean() * 100
            avg_neg = valid[valid['ensemble_signal'] < 0][col].mean() * 100
            print(f"Avg Return (Ensemble>0): {avg_pos:.3f}% | (Ensemble<0): {avg_neg:.3f}%")
