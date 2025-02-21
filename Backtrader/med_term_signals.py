import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats

class MediumTermTrendStrategy(bt.Strategy):
    """
    Collects MACD(12,26,9) and a 20-period EMA signals on *5-minute* resampled data.
    Measures forward returns (Information Coefficient, etc.).
    """

    params = (
        # MACD parameters
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),

        # EMA parameters
        ('ema_period', 20),

        # Forward return horizons
        ('forward_periods', [5, 15, 30]),  # e.g. 5-bars, 15-bars, 30-bars on 5m data

        # Signal thresholds
        ('macd_std_threshold', 1.5),     # Increased for stronger signals
        ('min_trend_strength', 0.003),   # Require stronger trend confirmation
        ('momentum_period', 5),          # Keep momentum check
        
        # Market hours (Eastern Time)
        ('market_open_hour', 9),
        ('market_open_min', 30),
        ('market_close_hour', 16),
        ('market_close_min', 0),
        
        # Volume filter
        ('volume_ma_period', 20),
        ('min_volume_mult', 1.2)  # Reduced from 1.5
    )

    def __init__(self):
        """Initialize 5-minute indicators and signal storage."""
        # Only use the 5-minute resampled data
        self.data5m = self.datas[1]  # resampled 5-min data

        # MACD on 5m data
        self.macd = bt.indicators.MACD(
            self.data5m,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal,
            plot=False
        )

        # 20-bar EMA on 5m data
        self.ema = bt.indicators.ExponentialMovingAverage(
            self.data5m,
            period=self.p.ema_period,
            plot=False
        )
        
        # Volume MA for filtering
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data5m.volume,
            period=self.p.volume_ma_period
        )

        # Store signals and bar indices
        self.signals = []
        self.signal_times = set()  # Track timestamps to prevent duplicates
        
        # For tracking MACD threshold
        self.macd_diffs = []
        
        # Track market hours bars
        self.market_hours_count = 0
        self.last_bar_time = None  # For tracking unique bars

    def is_market_hours(self, dt):
        """Check if time is within market hours (converts UTC to ET)."""
        # Convert UTC to ET (UTC-4 for EDT, UTC-5 for EST)
        # For simplicity, using UTC-4 (EDT)
        et_hour = (dt.hour - 4) % 24
        
        # Basic market hours check
        if et_hour < self.p.market_open_hour or et_hour > self.p.market_close_hour:
            return False
        if et_hour == self.p.market_open_hour and dt.minute < self.p.market_open_min:
            return False
        if et_hour == self.p.market_close_hour and dt.minute >= self.p.market_close_min:
            return False
            
        # Skip first 15 minutes after open and last 15 minutes before close
        if (et_hour == self.p.market_open_hour and 
            dt.minute < self.p.market_open_min + 15):
            return False
        if (et_hour == self.p.market_close_hour and 
            dt.minute >= self.p.market_close_min - 15):
            return False
            
        return True

    def next(self):
        """Compute MACD-based signal on 5-minute bars."""
        # Skip if indicators aren't ready
        if not self.macd.macd[0] or not self.macd.signal[0] or not self.ema[0]:
            return
            
        # Check for duplicate timestamp
        current_time = self.data5m.datetime.datetime(0)
        
        # Only process 5-minute data feed
        if self.data5m._name != self.datas[1]._name:
            return
            
        # Check market hours
        if not self.is_market_hours(current_time):
            return
            
        # Only count unique bars
        if current_time != self.last_bar_time:
            self.market_hours_count += 1
            self.last_bar_time = current_time
            
        if current_time in self.signal_times:
            return
            
        # Calculate MACD signal
        macd_diff = self.macd.macd[0] - self.macd.signal[0]
        self.macd_diffs.append(macd_diff)
        
        # Volume and momentum checks
        if len(self.volume_ma) > 0:
            # Volume check
            vol_ratio = self.data5m.volume[0] / self.volume_ma[0]
            if vol_ratio < self.p.min_volume_mult:
                return
                
            # Momentum check (5-bar return)
            if len(self.data5m) > self.p.momentum_period:
                momentum = (self.data5m.close[0] - self.data5m.close[-self.p.momentum_period]) / self.data5m.close[-self.p.momentum_period]
                # Momentum should be in same direction as signals
                if (momentum > 0) != (macd_diff > 0):
                    return
        
        # Wait for enough history
        if len(self.macd_diffs) < 20:
            return
            
        # Dynamic threshold based on recent volatility
        macd_threshold = np.std(self.macd_diffs[-20:]) * self.p.macd_std_threshold
        
        # Check MACD signal strength
        if abs(macd_diff) <= macd_threshold:
            return  # MACD signal too weak
            
        # MACD direction
        macd_signal = 1 if macd_diff > 0 else -1
            
        # Check EMA trend strength
        price_to_ema = (self.data5m.close[0] - self.ema[0]) / self.ema[0]
        if abs(price_to_ema) < self.p.min_trend_strength:
            return  # Trend too weak
            
        # EMA direction
        ema_signal = 1 if price_to_ema > 0 else -1
        
        # Generate signal if MACD and EMA agree
        if macd_signal == ema_signal:
            # Store signal with metadata
            current_time = self.data5m.datetime.datetime(0)
            self.signal_times.add(current_time)
            self.signals.append({
                'bar_index': len(self.data5m)-1,
                'datetime': current_time,
                'macd_signal': macd_signal,
                'ema_signal': ema_signal,
                'combined_signal': macd_signal,
                'close': self.data5m.close[0],
                'macd_diff': macd_diff,
                'macd_threshold': macd_threshold,
                'price_to_ema': price_to_ema,
                'volume_ratio': vol_ratio
            })

    def stop(self):
        """Compute forward returns on the 5-minute bars and measure correlation (IC)."""
        if not self.signals:
            print("No signals generated!")
            return

        df = pd.DataFrame(self.signals)
        close_array = np.array(self.data5m.close.array)
        
        print("\n=== Medium-Term Trend Analysis (5m Data) ===")
        print(f"Total 5m Bars: {len(self.data5m)}")
        print(f"Market Hours 5m Bars: {self.market_hours_count}")
        print(f"Signals Generated: {len(df)}")
        print(f"Signal Rate: {len(df)/self.market_hours_count*100:.1f}% of market hours bars")

        # Signal distribution stats
        print("\nSignal Distribution:")
        print(f"Bullish Signals: {len(df[df['combined_signal'] == 1])}")
        print(f"Bearish Signals: {len(df[df['combined_signal'] == -1])}")
        
        # MACD stats
        print("\nMACD Statistics:")
        print(f"Average Threshold: {df['macd_threshold'].mean():.6f}")
        print(f"Average |MACD Diff|: {abs(df['macd_diff']).mean():.6f}")

        for period in self.p.forward_periods:
            col_name = f'fwd_ret_{period}'
            fwd_returns = []
            for idx in df['bar_index']:
                # If there's enough data ahead:
                if idx + period < len(close_array):
                    ret = (close_array[idx + period] - close_array[idx]) / close_array[idx]
                else:
                    ret = np.nan
                fwd_returns.append(ret)

            df[col_name] = fwd_returns

            valid = df.dropna(subset=[col_name]).copy()
            if len(valid) < 2:
                print(f"\nNot enough valid signals for {period}-bar forward returns.")
                continue

            ic = valid['combined_signal'].corr(valid[col_name])

            # T-test for significance
            n = len(valid)
            if abs(ic) < 1.0:
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
            else:
                t_stat = np.inf
                p_val = 0.0

            print(f"\n--- {period}-Bar Forward Returns ---")
            print(f"Number of Valid 5m Bars: {n}")
            print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")
            
            # Average returns by signal type
            avg_ret_pos = valid[valid['combined_signal'] > 0][col_name].mean()
            avg_ret_neg = valid[valid['combined_signal'] < 0][col_name].mean()
            print(f"Avg Return (Sig>0): {avg_ret_pos*100:.3f}% | (Sig<0): {avg_ret_neg*100:.3f}%")

def run_medium_term_analysis(csv_path):
    """
    Main function to:
      1) load 1-min data
      2) resample to 5-min
      3) run MediumTermTrendStrategy
    """
    # Create a Cerebro engine
    cerebro = bt.Cerebro()

    # Load 1-minute data
    data1m = bt.feeds.GenericCSVData(
        dataname=csv_path,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        dtformat='%Y-%m-%d %H:%M:%S%z',
        timeframe=bt.TimeFrame.Minutes,
        compression=1
    )

    # Add data feeds in order:
    # 1. Original 1-min data
    cerebro.adddata(data1m)
    
    # 2. Resample to 5-min
    data5m = cerebro.resampledata(
        data1m,
        timeframe=bt.TimeFrame.Minutes,
        compression=5
    )

    # Add strategy
    cerebro.addstrategy(MediumTermTrendStrategy)

    # Run analysis
    cerebro.run()

if __name__ == '__main__':
    """Analyze all symbols."""
    import os
    
    # Get all CSV files from historical_data directory
    data_dir = 'data/historical_data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_1m_20231227_20241226.csv')]
    symbols = sorted([f.split('_')[0] for f in csv_files])
    
    print(f"Found {len(symbols)} symbols to analyze:")
    print(", ".join(symbols))
    print("\nStarting analysis...")
    
    # Run analysis for each symbol
    for symbol in symbols:
        print(f"\n{'='*20} Analyzing {symbol} {'='*20}")
        csv_file = f'{data_dir}/{symbol}_1m_20231227_20241226.csv'
        run_medium_term_analysis(csv_file)
