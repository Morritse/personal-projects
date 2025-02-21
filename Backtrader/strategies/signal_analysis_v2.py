import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta
from scipy import stats

class SignalAnalysisV2(bt.Strategy):
    """
    Pure signal analysis strategy with enhanced data quality checks
    and multiple indicator combinations
    """
    
    params = (
        # RSI Parameters
        ('rsi2_period', 2),        # Ultra-fast RSI
        ('rsi2_ob', 95),          # More extreme thresholds for RSI(2)
        ('rsi2_os', 5),
        ('rsi5_period', 5),        # Fast RSI
        ('rsi5_ob', 80),          # Standard thresholds for RSI(5)
        ('rsi5_os', 20),
        # Stochastic Parameters
        ('stoch_period', 5),
        ('stoch_period_d', 3),     
        ('stoch_period_k', 3),     
        ('stoch_ob', 85),  
        ('stoch_os', 15),
        # Volume Parameters
        ('volume_ma_period', 20),  # For volume spike detection
        ('min_volume_mult', 1.5),  # Minimum volume multiple of average
        # Analysis Parameters
        ('forward_periods', [1, 5, 15]),
        # Session Parameters - include all trading hours
        ('market_open', time(9, 30)),
        ('market_close', time(16, 0))
    )
    
    def __init__(self):
        """Initialize indicators and tracking variables."""
        # Data quality tracking
        self.total_bars = 0
        self.valid_bars = 0
        self.gaps_detected = 0
        self.last_timestamp = None
        self.daily_stats = {}  # {date: {'gaps': count, 'bars': count}}
        
        # Technical Indicators
        self.rsi2 = bt.indicators.RSI(
            self.data,
            period=self.p.rsi2_period,
            safediv=True,
            plotname='RSI2'
        )
        
        self.rsi5 = bt.indicators.RSI(
            self.data,
            period=self.p.rsi5_period,
            safediv=True,
            plotname='RSI5'
        )
        
        self.stoch = bt.indicators.Stochastic(
            self.data,
            period=self.p.stoch_period,
            period_dfast=self.p.stoch_period_d,
            period_dslow=self.p.stoch_period_k,
            safediv=True,
            plotname='Stochastic'
        )
        
        # Volume indicators
        self.volume_ma = bt.indicators.SMA(
            self.data.volume,
            period=self.p.volume_ma_period
        )
        
        # Store signals and analysis data
        self.signals = []
        
        # Track indicator distributions
        self.rsi2_dist = []
        self.rsi5_dist = []
        self.stoch_dist = []
        self.volume_ratios = []
        
    def is_valid_bar(self, current_time):
        """Check if current bar meets data quality criteria."""
        # Must be within market hours
        if not (self.p.market_open <= current_time.time() <= self.p.market_close):
            return False
            
        # Check volume
        if len(self) > self.p.volume_ma_period:
            volume_ratio = self.data.volume[0] / self.volume_ma[0]
            if volume_ratio < self.p.min_volume_mult:
                return False
            self.volume_ratios.append(volume_ratio)
            
        return True
        
    def get_rsi2_signal(self):
        """Get signal from ultra-fast RSI(2)."""
        try:
            if self.rsi2[0] > self.p.rsi2_ob:
                return -1  # Extremely overbought
            elif self.rsi2[0] < self.p.rsi2_os:
                return 1   # Extremely oversold
            return 0
        except:
            return 0
            
    def get_rsi5_signal(self):
        """Get signal from fast RSI(5)."""
        try:
            if self.rsi5[0] > self.p.rsi5_ob:
                return -1  # Overbought
            elif self.rsi5[0] < self.p.rsi5_os:
                return 1   # Oversold
            return 0
        except:
            return 0
            
    def get_stoch_signal(self):
        """Get signal from Stochastic."""
        try:
            if self.stoch.percK[0] > self.p.stoch_ob:
                return -1  # Overbought
            elif self.stoch.percK[0] < self.p.stoch_os:
                return 1   # Oversold
            return 0
        except:
            return 0
            
    def next(self):
        """Process next bar and collect signal data."""
        # Data quality checks
        self.total_bars += 1
        current_time = self.data.datetime.datetime(0)
        
        # Track daily statistics
        current_date = current_time.date()
        if current_date not in self.daily_stats:
            self.daily_stats[current_date] = {'gaps': 0, 'bars': 0, 'valid_bars': 0}
            
        # Check for gaps
        if self.last_timestamp:
            expected_diff = timedelta(minutes=1)
            actual_diff = current_time - self.last_timestamp
            if actual_diff > expected_diff:
                self.gaps_detected += 1
                self.daily_stats[current_date]['gaps'] += 1
                
        self.last_timestamp = current_time
        self.daily_stats[current_date]['bars'] += 1
        
        # Check if bar is valid for analysis
        if not self.is_valid_bar(current_time):
            return
            
        self.valid_bars += 1
        self.daily_stats[current_date]['valid_bars'] += 1
        
        # Track indicator distributions
        self.rsi2_dist.append(self.rsi2[0])
        self.rsi5_dist.append(self.rsi5[0])
        self.stoch_dist.append(self.stoch.percK[0])
        
        # Get signals from each indicator
        rsi2_signal = self.get_rsi2_signal()
        rsi5_signal = self.get_rsi5_signal()
        stoch_signal = self.get_stoch_signal()
        
        # Calculate forward returns
        returns = {}
        for period in self.p.forward_periods:
            if len(self) < len(self.data) - period:  # Check if we have enough future data
                future_price = self.data.close[period]  # Price N bars ahead
                fwd_return = (future_price - self.data.close[0]) / self.data.close[0] * 100
                returns[f'fwd_ret_{period}'] = fwd_return
            else:
                returns[f'fwd_ret_{period}'] = None
                
        # Store signal data
        signal_data = {
            'datetime': current_time,
            'rsi2_signal': rsi2_signal,
            'rsi5_signal': rsi5_signal,
            'stoch_signal': stoch_signal,
            'price': self.data.close[0],
            'volume': self.data.volume[0],
            'volume_ratio': self.data.volume[0] / self.volume_ma[0] if len(self) > self.p.volume_ma_period else None,
            'bar_index': len(self),
            'hour': current_time.hour,
            'minute': current_time.minute,
            **returns  # Add forward returns to signal data
        }
        
        self.signals.append(signal_data)
        
    def stop(self):
        """Calculate and print analysis metrics."""
        if not self.signals:
            return
            
        print("\n=== Data Quality Analysis ===")
        print(f"Total Bars: {self.total_bars}")
        print(f"Valid Bars: {self.valid_bars}")
        print(f"Gaps Detected: {self.gaps_detected}")
        coverage = (self.valid_bars / self.total_bars) * 100
        print(f"Valid Bar Coverage: {coverage:.1f}%")
        
        print("\n=== Daily Statistics ===")
        good_days = 0
        total_days = len(self.daily_stats)
        for date, stats in sorted(self.daily_stats.items()):
            gap_ratio = stats['gaps'] / stats['bars'] if stats['bars'] > 0 else 1
            valid_ratio = stats['valid_bars'] / stats['bars'] if stats['bars'] > 0 else 0
            if gap_ratio < 0.1 and valid_ratio > 0.8:  # Good quality day criteria
                good_days += 1
            print(f"{date}: {stats['bars']} bars, {stats['gaps']} gaps, {stats['valid_bars']} valid ({valid_ratio*100:.1f}%)")
            
        print(f"\nGood Quality Days: {good_days}/{total_days} ({good_days/total_days*100:.1f}%)")
        
        # Convert signals to DataFrame
        signals_df = pd.DataFrame(self.signals)
        
        # Calculate forward returns
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
        
        # Filter to good quality days
        good_dates = [date for date, stats in self.daily_stats.items() 
                     if stats['gaps'] / stats['bars'] < 0.1 and 
                     stats['valid_bars'] / stats['bars'] > 0.8]
        signals_df = signals_df[signals_df['datetime'].dt.date.isin(good_dates)]
        
        print(f"\nAnalyzing {len(signals_df)} bars from {len(good_dates)} good quality days")
        
        # Analyze each indicator separately
        for indicator in ['rsi2', 'rsi5', 'stoch']:
            print(f"\n=== {indicator.upper()} Analysis ===")
            signal_col = f'{indicator}_signal'
            
            # Signal distribution
            signals = signals_df[signal_col].value_counts()
            print("\nSignal Distribution:")
            print(f"Oversold (1): {signals.get(1, 0)}")
            print(f"Neutral (0): {signals.get(0, 0)}")
            print(f"Overbought (-1): {signals.get(-1, 0)}")
            
            # Calculate ICs for different horizons
            for period in self.p.forward_periods:
                ret_col = f'fwd_ret_{period}'
                valid_signals = signals_df[
                    (signals_df[signal_col] != 0) & 
                    (signals_df[ret_col].notna())
                ]
                
                if len(valid_signals) > 0:
                    ic = valid_signals[signal_col].corr(valid_signals[ret_col])
                    n = len(valid_signals)
                    t_stat = ic * np.sqrt((n-2)/(1-ic**2))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
                    
                    print(f"\n{period}-Minute Forward Returns:")
                    print(f"IC: {ic:.3f} (p-value: {p_value:.3f}, n={n})")
                    
                    # Average returns by signal type
                    avg_ret_buy = valid_signals[valid_signals[signal_col] == 1][ret_col].mean()
                    avg_ret_sell = valid_signals[valid_signals[signal_col] == -1][ret_col].mean()
                    print(f"Avg Return after Oversold: {avg_ret_buy:.3f}%")
                    print(f"Avg Return after Overbought: {avg_ret_sell:.3f}%")
                    
                    # Time-of-day analysis
                    print("\nTime-of-Day Breakdown:")
                    hourly_ic = valid_signals.groupby('hour').apply(
                        lambda x: x[signal_col].corr(x[ret_col]) if len(x) > 10 else np.nan
                    ).dropna()
                    
                    print("\nHour | IC    | Signals")
                    print("-" * 25)
                    for hour in sorted(hourly_ic.index):
                        n_signals = len(valid_signals[valid_signals['hour'] == hour])
                        print(f"{hour:02d}:00 | {hourly_ic[hour]:.3f} | {n_signals:7d}")
