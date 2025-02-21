# long_term_trend_analysis.py

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import pytz
import os

class LongTermTrendStrategy(bt.Strategy):
    """
    Long-term trend approach on 15-min resampled data:
      - 200-bar SMA to determine direction: +1 if close > SMA, else -1
      - ADX(14) to measure trend strength:
          * If ADX < 20 => scale signal by 0.5 (range-bound)
          * If ADX > 25 => full signal (strong trend)
      - We measure forward returns (IC) at multiple horizons (e.g., 15, 30, 60 bars).
    """

    params = (
        # Trend indicators
        ('sma_period', 100),    # "100-bar" SMA on 15-min
        ('adx_period', 14),
        ('adx_low', 20),        # threshold for range-bound
        ('adx_high', 30),       # threshold for strong trend
        
        # Forward returns
        ('forward_periods', [15, 30, 60, 90]),  # measure forward returns
        
        # Market hours (Eastern Time)
        ('market_open_hour', 9),
        ('market_open_min', 30),
        ('market_close_hour', 16),
        ('market_close_min', 0),
        ('skip_open_minutes', 10),   # Skip first 10 min
        ('skip_close_minutes', 10),  # Skip last 10 min
        
        # Volume filter
        ('volume_ma_period', 20),
        ('min_volume_mult', 1.0)  # Current volume must be at least average
    )


    def __init__(self):
        # Use the 15-min resampled feed
        self.data15 = self.datas[1]  # data15m is the second feed

        # 200-bar SMA
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data15,
            period=self.p.sma_period,
            plot=False
        )

        # ADX(14)
        self.adx = bt.indicators.ADX(
            self.data15,
            period=self.p.adx_period,
            plot=False
        )
        
        # Volume MA
        self.volume_ma = bt.indicators.SimpleMovingAverage(
            self.data15.volume,
            period=self.p.volume_ma_period,
            plot=False
        )

        # We'll store signals per bar
        self.signals = []
        self.signal_times = set()  # Prevent duplicates
        
        # For timezone conversion
        self.ny_tz = pytz.timezone("US/Eastern")

    def is_market_hours(self, dt_utc):
        """Check if time is within market hours (converts UTC to ET)."""
        # Make dt_utc timezone-aware
        dt_utc_aware = dt_utc.replace(tzinfo=pytz.UTC)
        # Convert to Eastern Time
        dt_et = dt_utc_aware.astimezone(self.ny_tz)

        hour = dt_et.hour
        minute = dt_et.minute

        # Basic market hours check
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
        """
        Compute 'long-term' signal each 15-min bar:
          +1 if close > SMA, else -1
          Then scale it if ADX < 15 => *0.5, if ADX >25 => *1.0, else *0.75
        """
        # Only process when we're on the 15-min data feed
        if self.data != self.data15:
            return
            
        # Get current time
        current_time = self.data15.datetime.datetime(0)
        
        # Skip if outside market hours
        if not self.is_market_hours(current_time):
            return
            
        # Skip if already processed this timestamp
        if current_time in self.signal_times:
            return
            
        # Skip if indicators aren't ready
        if not self.sma[0] or not self.adx[0] or not self.volume_ma[0]:
            return
            
        # Volume check
        if self.volume_ma[0] != 0:  # Prevent division by zero
            vol_ratio = self.data15.volume[0] / self.volume_ma[0]
            if vol_ratio < self.p.min_volume_mult:
                return
            
        # Direction
        if self.data15.close[0] > self.sma[0]:
            base_signal = 1
        else:
            base_signal = -1

        # ADX-based scaling
        adx_val = self.adx[0]
        if adx_val < self.p.adx_low:
            # Range-bound => partial signal
            final_signal = 0.5 * base_signal
        elif adx_val > self.p.adx_high:
            # Strong trend => full signal
            final_signal = 1.0 * base_signal
        else:
            # In-between (20 < ADX < 25) => partial scale, e.g., 0.75
            final_signal = 0.75 * base_signal
            
            # Track this timestamp
        self.signal_times.add(current_time)
        
        # Store signal with metadata
        self.signals.append({
            'bar_index': len(self.data15) - 1,  # current bar index
            'datetime': current_time,
            'base_signal': base_signal,
            'adx_val': adx_val,
            'final_signal': final_signal,
            'close': self.data15.close[0],
            'volume_ratio': vol_ratio
        })

    def stop(self):
        """Compute forward returns, measure correlation (IC)."""
        if not self.signals:
            print("No signals generated!")
            return

        df = pd.DataFrame(self.signals)
        close_array = np.array(self.data15.close.array)

        print("\n=== Long-Term Trend Analysis (15m Data) ===")
        print(f"Total 15m Bars: {len(self.data15)}")
        print(f"Signal Rows: {len(df)}")
        print(f"Signal Rate: {len(df)/len(self.data15)*100:.1f}% of 15m bars")

        for period in self.p.forward_periods:
            col_name = f'fwd_ret_{period}'
            returns_list = []

            for idx in df['bar_index']:
                if idx + period < len(close_array):
                    ret = (close_array[idx + period] - close_array[idx]) / close_array[idx]
                else:
                    ret = np.nan
                returns_list.append(ret)

            df[col_name] = returns_list
            valid = df.dropna(subset=[col_name])

            if len(valid) < 2:
                print(f"\nNot enough valid signals for {period}-bar forward returns.")
                continue

            # Correlate final_signal with forward returns
            ic = valid['final_signal'].corr(valid[col_name])

            # T-test
            n = len(valid)
            if abs(ic) < 1.0:
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
            else:
                t_stat = np.inf
                p_val = 0.0

            print(f"\n--- {period}-Bar Forward Returns ---")
            print(f"Number of Valid 15m Bars: {n}")
            print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")

            # Optional: average forward return for final_signal>0 vs. <0
            avg_ret_pos = valid[valid['final_signal'] > 0][col_name].mean()
            avg_ret_neg = valid[valid['final_signal'] < 0][col_name].mean()
            print(f"Avg Ret (Sig>0): {avg_ret_pos*100:.3f}% | (Sig<0): {avg_ret_neg*100:.3f}%\n")


def run_long_term_analysis(csv_path):
    """
    1) Load 1-min data from CSV
    2) Resample to 15-min
    3) Apply the LongTermTrendStrategy
    4) Print IC, p-values, etc.
    """
    cerebro = bt.Cerebro()

    # 1-minute feed
    data1m = bt.feeds.GenericCSVData(
        dataname=csv_path,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        dtformat='%Y-%m-%d %H:%M:%S%z',  # adjust if needed
        timeframe=bt.TimeFrame.Minutes,
        compression=1
    )

    cerebro.adddata(data1m, name='data1m')

    # Resample to 15m
    data15m = cerebro.resampledata(
        data1m,
        timeframe=bt.TimeFrame.Minutes,
        compression=15,
        name='data15m'
    )

    cerebro.addstrategy(LongTermTrendStrategy)
    cerebro.run()

    # (Optional) Print bar counts:
    print("\n=== Bar Counts ===")
    print(f"1m bars: {len(data1m)}")
    print(f"15m bars: {len(data15m)}")


if __name__ == '__main__':
    """Analyze all symbols with long-term strategy."""
    # Get all CSV files from historical_data directory
    data_dir = 'data/historical_data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_1m_20231227_20241226.csv')]
    symbols = sorted([f.split('_')[0] for f in csv_files])
    
    print(f"Found {len(symbols)} symbols to analyze:")
    print(", ".join(symbols))
    print("\nStarting long-term analysis...")
    
    # Run analysis for each symbol
    for symbol in symbols:
        print(f"\n{'='*20} Analyzing {symbol} {'='*20}")
        csv_file = f'{data_dir}/{symbol}_1m_20231227_20241226.csv'
        run_long_term_analysis(csv_file)
