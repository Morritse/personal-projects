import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import pytz

class SignalAnalysisStrategy(bt.Strategy):
    """
    Strategy that ONLY collects RSI/Stoch signals within US/Eastern market hours
    and later computes forward returns (IC) in the stop() method. No actual trades.
    """

# --- Short-Term RSI/Stoch Parameters ---
    params = (
        # RSI: Very short to catch quick reversion
        ('rsi_period', 2),
        ('rsi_overbought', 90),
        ('rsi_oversold', 10),

        # Stoch: Faster than default, also tight thresholds
        ('stoch_period', 5),
        ('stoch_period_d', 2),
        ('stoch_period_k', 2),
        ('stoch_overbought', 90),
        ('stoch_oversold', 10),

        # Forward returns to evaluate short intraday reversion
        ('forward_periods', [1, 3, 10]),

        # Market hours (Eastern Time) – optional skip of first/last 15 min
        ('market_open_hour', 9),
        ('market_open_min', 30),
        ('market_close_hour', 16),
        ('market_close_min', 0),
        ('skip_open_minutes', 15),
        ('skip_close_minutes', 15),
    )



    def __init__(self):
        """Initialize indicators and signal storage."""
        # RSI & Stochastic on self.data
        self.rsi = bt.indicators.RSI(
            self.data,
            period=self.p.rsi_period,
            safediv=True
        )
        self.stoch = bt.indicators.Stochastic(
            self.data,
            period=self.p.stoch_period,
            period_dfast=self.p.stoch_period_d,
            period_dslow=self.p.stoch_period_k,
            safediv=True
        )

        # We'll store signals and bar indices here
        self.signals = []

        # Timezone for Eastern
        self.ny_tz = pytz.timezone("US/Eastern")

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
        """Compute signals each bar (if within market hours) and store them, but do not trade."""
        # Get current bar's naive-UTC datetime
        current_time_utc = self.data.datetime.datetime(0)

        # Filter out bars outside Eastern market hours (if desired)
        if not self.is_market_hours(current_time_utc):
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
        # (Here we do a simple approach: only emit a signal if RSI & Stoch match)
        combined_sig = 0
        if rsi_sig == stoch_sig and rsi_sig != 0:
            combined_sig = rsi_sig

        # Store the signal with bar index
        self.signals.append({
            'bar_index': len(self),
            'datetime': current_time_utc,
            'signal': combined_sig,
            'close': self.data.close[0],
        })

    def stop(self):
        """
        After all bars have run, compute forward returns for each signal
        and measure correlation (Information Coefficient).
        """
        if not self.signals:
            print("No signals generated!")
            return

        df = pd.DataFrame(self.signals)
        close_array = np.array(self.data.close.array)

        print("\n=== Signal Analysis ===")
        print(f"Total Bars Analyzed: {len(self)}")
        print(f"Signal Rows: {len(df)}")
        print(f"Oversold Signals: {len(df[df['signal'] == 1])}")
        print(f"Overbought Signals: {len(df[df['signal'] == -1])}")

        # For each desired forward period, compute returns & correlation
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

            # Filter valid signals (non-NaN returns, non-zero signal)
            valid = df.dropna(subset=[col_name])
            valid = valid[valid['signal'] != 0]
            if len(valid) < 2:
                print(f"\nNot enough valid signals for {period}-bar forward returns.")
                continue

            # Correlation (IC) between signal and forward returns
            ic = valid['signal'].corr(valid[col_name])

            # T-test for significance
            n = len(valid)
            if abs(ic) < 1.0:  # avoid domain errors if ic ~ ±1
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
            else:
                t_stat = np.inf
                p_val = 0.0

            print(f"\n--- {period}-Bar Forward Returns ---")
            print(f"Number of Valid Signals: {n}")
            print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")

            # Average forward return after oversold vs. overbought
            avg_ret_oversold = valid[valid['signal'] == 1][col_name].mean()
            avg_ret_overbought = valid[valid['signal'] == -1][col_name].mean()
            print(f"Avg Return after Oversold: {avg_ret_oversold*100:.3f}%")
            print(f"Avg Return after Overbought: {avg_ret_overbought*100:.3f}%")


def run_signal_analysis(csv_path):
    """
    Main function to run the signal analysis on a single CSV file.
    """
    # 1) Create a Cerebro engine
    cerebro = bt.Cerebro()

    # 2) Define your data feed
    data = bt.feeds.GenericCSVData(
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

    # 3) Add the data feed
    cerebro.adddata(data)

    # 4) Add our analysis strategy
    cerebro.addstrategy(SignalAnalysisStrategy)

    # 5) Run it
    cerebro.run()


if __name__ == '__main__':
    """Analyze all symbols in historical_data."""
    import os
    
    data_dir = 'data/historical_data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_1m_20231227_20241226.csv')]
    symbols = sorted([f.split('_')[0] for f in csv_files])
    
    print(f"Found {len(symbols)} symbols to analyze:")
    print(", ".join(symbols))
    print("\nStarting analysis...")
    
    for symbol in symbols:
        print(f"\n{'='*20} Analyzing {symbol} {'='*20}")
        csv_file = f'{data_dir}/{symbol}_1m_20231227_20241226.csv'
        run_signal_analysis(csv_file)
