import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from scipy import stats

class SignalAnalysis(bt.Strategy):
    """
    Pure signal analysis strategy - no trading, just measuring predictive power
    of RSI and Stochastic indicators at different horizons
    """
    
    params = (
        # Indicator Parameters
        ('rsi_period', 5),         # Faster RSI
        ('stoch_period', 5),       # Faster Stochastic
        ('stoch_period_d', 3),     
        ('stoch_period_k', 3),     
        ('rsi_overbought', 80),    # Adjusted to ~90th percentile
        ('rsi_oversold', 20),      # Adjusted to ~10th percentile
        ('stoch_overbought', 87),  # Adjusted to ~90th percentile
        ('stoch_oversold', 13),    # Adjusted to ~10th percentile
        # Analysis Parameters
        ('forward_periods', [1, 5, 15]),  # Forward return horizons to analyze
        ('market_open', time(9, 30)),
        ('market_close', time(16, 0)),
        ('warmup_minutes', 15),    # Skip first 15 min
        ('cooldown_minutes', 15),  # Skip last 15 min
    )
    
    def __init__(self):
        """Initialize indicators and tracking variables."""
        # Data quality tracking
        self.total_bars = 0
        self.market_hours_bars = 0
        self.gaps_detected = 0
        self.last_timestamp = None
        self.rsi_dist = []  # Track RSI distribution
        self.stoch_dist = []  # Track Stochastic distribution
        
        # Track daily statistics
        self.daily_stats = {}  # {date: {'gaps': count, 'bars': count}}
        
        # Track time-of-day statistics
        self.hour_stats = {}  # {hour: {'signals': 0, 'wins': 0, 'total_return': 0.0}}
        for hour in range(9, 17):  # 9:30 AM to 4:00 PM
            self.hour_stats[hour] = {'signals': 0, 'wins': 0, 'total_return': 0.0}
        
        # Technical Indicators
        self.rsi = bt.indicators.RSI(
            self.data,
            period=self.p.rsi_period,
            safediv=True,
            plotname='RSI'
        )
        
        # Add faster RSI(2) for comparison
        self.rsi2 = bt.indicators.RSI(
            self.data,
            period=2,
            safediv=True,
            plotname='RSI2'
        )
        
        self.stoch = bt.indicators.Stochastic(
            self.data,
            period=self.p.stoch_period,
            period_dfast=self.p.stoch_period_d,
            period_dslow=self.p.stoch_period_k,
            safediv=True,
            plotname='Stochastic'
        )
        
        # Track RSI(2) distribution
        self.rsi2_dist = []
        
        # Moving average for trend identification
        self.sma = bt.indicators.SMA(
            self.data,
            period=20,
            plotname='20 SMA'
        )
        
        # Store signals and returns for analysis
        self.signals = []
        
        # Track market regimes
        self.vol_window = []  # Rolling volatility calculation
        
    def is_market_hours(self, dt):
        """Check if timestamp is within valid market hours."""
        t = dt.time()
        
        # Must be within market hours
        if not (self.p.market_open <= t <= self.p.market_close):
            return False
            
        # Skip warmup period after open
        if t < (datetime.combine(dt.date(), self.p.market_open) + 
               timedelta(minutes=self.p.warmup_minutes)).time():
            return False
            
        # Skip cooldown period before close
        if t > (datetime.combine(dt.date(), self.p.market_close) - 
               timedelta(minutes=self.p.cooldown_minutes)).time():
            return False
            
        return True
        
    def get_combined_signal(self):
        """Get combined signal from RSI and Stochastic."""
        try:
            # RSI signal
            if self.rsi[0] > self.p.rsi_overbought:
                rsi_signal = -1  # Overbought
            elif self.rsi[0] < self.p.rsi_oversold:
                rsi_signal = 1   # Oversold
            else:
                rsi_signal = 0   # Neutral
                
            # Stochastic signal
            if self.stoch.percK[0] > self.p.stoch_overbought:
                stoch_signal = -1  # Overbought
            elif self.stoch.percK[0] < self.p.stoch_oversold:
                stoch_signal = 1   # Oversold
            else:
                stoch_signal = 0   # Neutral
                
            # Only return signal when both agree
            if rsi_signal == stoch_signal and rsi_signal != 0:
                return rsi_signal
            return 0
            
        except:
            return 0
            
    def get_market_regime(self):
        """Determine current market regime using past data only."""
        try:
            # Volatility regime
            if len(self.vol_window) >= 20:
                current_vol = np.std(self.vol_window[-20:])
                median_vol = np.median([abs(r) for r in self.vol_window[-20:]])
                high_vol = current_vol > median_vol
            else:
                high_vol = False
                
            # Trend regime
            if len(self) > 1:
                trending = abs(self.data.close[0] - self.sma[0]) / self.sma[0] > 0.01
            else:
                trending = False
                
            return {
                'high_vol': high_vol,
                'trending': trending
            }
            
        except:
            return {
                'high_vol': False,
                'trending': False
            }
    
    def next(self):
        """Store signals and calculate forward returns."""
        # Data quality checks
        self.total_bars += 1
        current_time = self.data.datetime.datetime(0)
        
        # Track daily statistics
        current_date = current_time.date()
        if current_date not in self.daily_stats:
            self.daily_stats[current_date] = {'gaps': 0, 'bars': 0}
            
        # Check for gaps during market hours
        if self.last_timestamp and self.is_market_hours(current_time):
            expected_diff = timedelta(minutes=1)
            actual_diff = current_time - self.last_timestamp
            if actual_diff > expected_diff:
                # Only log gaps during market hours
                market_open = datetime.combine(current_time.date(), self.p.market_open)
                market_close = datetime.combine(current_time.date(), self.p.market_close)
                if market_open <= current_time <= market_close:
                    self.gaps_detected += 1
                    self.daily_stats[current_date]['gaps'] += 1
                    
        self.last_timestamp = current_time
        
        # Update daily bar count
        if self.is_market_hours(current_time):
            self.daily_stats[current_date]['bars'] += 1
        
        # Skip if outside market hours
        if not self.is_market_hours(current_time):
            return
            
        self.market_hours_bars += 1
        
        # Track indicator distributions
        self.rsi_dist.append(self.rsi[0])
        self.rsi2_dist.append(self.rsi2[0])
        self.stoch_dist.append(self.stoch.percK[0])
        
        # Update volatility window
        if len(self) > 1:
            ret = (self.data.close[0] - self.data.close[-1]) / self.data.close[-1]
            self.vol_window.append(ret)
            
        # Get current signal and regime
        signal = self.get_combined_signal()
        if signal == 0:  # Only store actual signals
            return
            
        regime = self.get_market_regime()
        
        # Store signal data
        signal_data = {
            'datetime': current_time,
            'signal': signal,
            'price': self.data.close[0],
            'rsi': self.rsi[0],
            'stoch_k': self.stoch.percK[0],
            'high_vol': regime['high_vol'],
            'trending': regime['trending'],
            'bar_index': len(self),  # Store current bar index
            'hour': current_time.hour,
            'minute': current_time.minute
        }
        
        self.signals.append(signal_data)
    
    def stop(self):
        """Calculate and print signal analysis metrics."""
        if not self.signals:
            return
            
        print("\n=== Data Quality Analysis ===")
        print(f"Total Bars: {self.total_bars}")
        print(f"Market Hours Bars: {self.market_hours_bars}")
        print(f"Market Hours Gaps: {self.gaps_detected}")
        coverage = (self.market_hours_bars / self.total_bars) * 100
        print(f"Market Hours Coverage: {coverage:.1f}%")
        
        print("\n=== Daily Statistics ===")
        good_days = 0
        total_days = len(self.daily_stats)
        for date, stats in sorted(self.daily_stats.items()):
            gap_ratio = stats['gaps'] / stats['bars'] if stats['bars'] > 0 else 1
            if gap_ratio < 0.1:  # Less than 10% gaps
                good_days += 1
            print(f"{date}: {stats['bars']} bars, {stats['gaps']} gaps ({gap_ratio*100:.1f}% gaps)")
        print(f"\nDays with <10% gaps: {good_days}/{total_days} ({good_days/total_days*100:.1f}%)")
            
        print("\n=== Indicator Statistics ===")
        rsi_array = np.array(self.rsi_dist)
        rsi2_array = np.array(self.rsi2_dist)
        stoch_array = np.array(self.stoch_dist)
        
        print("\nRSI(5) Distribution:")
        print(f"Mean: {np.mean(rsi_array):.1f}")
        print(f"Std: {np.std(rsi_array):.1f}")
        print("Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"{p}th: {np.percentile(rsi_array, p):.1f}")
            
        print("\nRSI(2) Distribution:")
        print(f"Mean: {np.mean(rsi_array):.1f}")
        print(f"Std: {np.std(rsi_array):.1f}")
        print("Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"{p}th: {np.percentile(rsi_array, p):.1f}")
            
        print("\nStochastic Distribution:")
        print(f"Mean: {np.mean(stoch_array):.1f}")
        print(f"Std: {np.std(stoch_array):.1f}")
        print("Percentiles:")
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"{p}th: {np.percentile(stoch_array, p):.1f}")
        
        # Convert signals to DataFrame
        signals_df = pd.DataFrame(self.signals)
        
        # Calculate forward returns for each signal
        close_array = np.array(self.data.close.array)
        for period in self.p.forward_periods:
            returns = []
            for idx in signals_df['bar_index']:
                if idx + period < len(close_array):
                    fwd_return = (close_array[idx + period] - close_array[idx]) / close_array[idx] * 100
                    returns.append(fwd_return)
                else:
                    returns.append(np.nan)
            signals_df[f'fwd_ret_{period}'] = returns
        
        print('\n=== Signal Analysis ===')
        print(f'Total Signals: {len(signals_df)}')
        print(f'Oversold Signals: {len(signals_df[signals_df["signal"] == 1])}')
        print(f'Overbought Signals: {len(signals_df[signals_df["signal"] == -1])}')
        
        # Filter to good quality days
        good_dates = [date for date, stats in self.daily_stats.items() 
                     if stats['gaps'] / stats['bars'] < 0.1]
        signals_df = signals_df[signals_df['datetime'].dt.date.isin(good_dates)]
        
        print(f"\nAnalyzing {len(signals_df)} signals from {len(good_dates)} good quality days")
        
        # Time-of-day analysis
        print("\n=== Time-of-Day Analysis ===")
        signals_df['hour_minute'] = signals_df['datetime'].apply(lambda x: f"{x.hour:02d}:{x.minute:02d}")
        signals_df['win'] = signals_df.apply(
            lambda row: (row['signal'] * row['fwd_ret_1'] > 0), axis=1
        )
        
        # Group by hour
        hourly_stats = signals_df.groupby('hour').agg({
            'signal': 'count',
            'win': ['sum', 'mean'],
            'fwd_ret_1': 'mean'
        }).round(3)
        
        print("\nHourly Performance:")
        print("Hour | Signals | Win Rate | Avg Return")
        print("-" * 40)
        for hour, stats in hourly_stats.iterrows():
            signals = stats[('signal', 'count')]
            win_rate = stats[('win', 'mean')] * 100
            avg_ret = stats[('fwd_ret_1', 'mean')]
            print(f"{hour:02d}:00 | {signals:7d} | {win_rate:7.1f}% | {avg_ret:9.3f}%")
        
        # Calculate ICs for different horizons
        print('\nInformation Coefficients:')
        for period in self.p.forward_periods:
            print(f"\n{period}-Minute Forward Returns:")
            ret_col = f'fwd_ret_{period}'
            # Filter valid signals with returns
            valid_signals = signals_df[signals_df[ret_col].notna()].copy()
            print(f"\nValid signals for {period}-min returns: {len(valid_signals)}")
            if len(valid_signals) > 0:
                print("Sample of signals and returns:")
                print(valid_signals[['datetime', 'signal', ret_col]].head())
                
                # Calculate IC (correlation between signal and forward return)
                ic = valid_signals['signal'].corr(valid_signals[ret_col])
                
                # Calculate t-stat and p-value
                n = len(valid_signals)
                t_stat = ic * np.sqrt((n-2)/(1-ic**2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                
                print(f'\n{period}-Minute Forward Returns:')
                print(f'Overall IC: {ic:.3f} (p-value: {p_value:.3f})')
                print(f'Number of Signals: {len(valid_signals)}')
                
                # Average returns by signal type
                avg_ret_buy = valid_signals[valid_signals['signal'] == 1][ret_col].mean()
                avg_ret_sell = valid_signals[valid_signals['signal'] == -1][ret_col].mean()
                print(f'Avg Return after Oversold: {avg_ret_buy:.3f}%')
                print(f'Avg Return after Overbought: {avg_ret_sell:.3f}%')
                
                # IC by regime
                print('\nRegime Breakdown:')
                # High Volatility
                high_vol = valid_signals[valid_signals['high_vol']]
                if len(high_vol) > 0:
                    ic_hv = high_vol['signal'].corr(high_vol[ret_col])
                    print(f'High Vol IC: {ic_hv:.3f} (n={len(high_vol)})')
                    
                # Low Volatility
                low_vol = valid_signals[~valid_signals['high_vol']]
                if len(low_vol) > 0:
                    ic_lv = low_vol['signal'].corr(low_vol[ret_col])
                    print(f'Low Vol IC: {ic_lv:.3f} (n={len(low_vol)})')
                    
                # Trending
                trending = valid_signals[valid_signals['trending']]
                if len(trending) > 0:
                    ic_trend = trending['signal'].corr(trending[ret_col])
                    print(f'Trending IC: {ic_trend:.3f} (n={len(trending)})')
                    
                # Ranging
                ranging = valid_signals[~valid_signals['trending']]
                if len(ranging) > 0:
                    ic_range = ranging['signal'].corr(ranging[ret_col])
                    print(f'Ranging IC: {ic_range:.3f} (n={len(ranging)})')
