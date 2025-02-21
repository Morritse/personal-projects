import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
import pytz

class OnBalanceVolume(bt.Indicator):
    """
    On-Balance Volume (OBV) Indicator
    Formula:
      OBV(i) = OBV(i-1) + Volume(i)    if Close(i) > Close(i-1)
               OBV(i-1) - Volume(i)    if Close(i) < Close(i-1)
               OBV(i-1)                otherwise
    """
    lines = ('obv',)
    params = dict()

    def __init__(self):
        # Initialize OBV at zero
        self.l.obv[0] = 0

    def next(self):
        prev_close = self.data.close[-1]
        cur_close = self.data.close[0]
        prev_obv = self.l.obv[-1]  # previous OBV value
        volume = self.data.volume[0]

        if cur_close > prev_close:
            self.l.obv[0] = prev_obv + volume
        elif cur_close < prev_close:
            self.l.obv[0] = prev_obv - volume
        else:
            self.l.obv[0] = prev_obv

class VolumeVolatilityStrategy(bt.Strategy):
    """
    Volume & Volatility analysis on 5-min data:
    - OBV trend for volume flow direction
    - ATR for volatility regime
    - Combine into scaling factors
    """
    params = dict(
        # OBV parameters
        obv_lookback=1,      # Compare current OBV to N bars ago
        obv_scale_up=1.05,   # Multiplier when OBV rising
        obv_scale_down=0.95, # Multiplier when OBV falling
        
        # ATR parameters
        atr_period=14,       # ATR calculation period
        atr_ma_period=50,    # ATR moving average period
        atr_scale_min=0.8,   # Min scaling factor
        atr_scale_max=1.2,   # Max scaling factor
        
        # Forward returns to analyze
        forward_periods=[5, 15, 30],
        
        # Market hours (Eastern Time)
        market_open_hour=9,
        market_open_min=30,
        market_close_hour=16,
        market_close_min=0,
        skip_open_minutes=15,
        skip_close_minutes=15,
    )

    def __init__(self):
        """Initialize indicators on 5-min data."""
        self.data5m = self.datas[1]  # 5-min feed
        
        # Volume flow indicator
        self.obv = OnBalanceVolume(self.data5m)
        
        # Volatility indicators
        self.atr = bt.indicators.ATR(
            self.data5m,
            period=self.p.atr_period,
            plot=False
        )
        self.atr_ma = bt.indicators.SimpleMovingAverage(
            self.atr,
            period=self.p.atr_ma_period,
            plot=False
        )
        
        # Store signals with factors
        self.signals = []
        
        # For timezone conversion
        self.ny_tz = pytz.timezone("US/Eastern")

    def is_market_hours(self, dt_utc):
        """Check if time is within market hours (converts UTC to ET)."""
        dt_utc_aware = dt_utc.replace(tzinfo=pytz.UTC)
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
        """Process 5-min bars to generate volume/volatility factors."""
        # Only process 5-min data
        if self.data != self.data5m:
            return
            
        # Get current time
        dt_utc = self.data5m.datetime.datetime(0)
        if not self.is_market_hours(dt_utc):
            return
            
        # Wait for indicators
        if not self.obv[0] or not self.atr[0] or not self.atr_ma[0]:
            return
            
        # 1) Volume Flow Factor
        vol_flow_factor = 1.0  # neutral default
        if len(self.obv) > self.p.obv_lookback:
            obv_change = self.obv[0] - self.obv[-self.p.obv_lookback]
            if obv_change > 0:
                vol_flow_factor = self.p.obv_scale_up
            elif obv_change < 0:
                vol_flow_factor = self.p.obv_scale_down
                
        # 2) Volatility Scaling Factor
        vol_scaler = 1.0  # neutral default
        if self.atr_ma[0] != 0:  # prevent division by zero
            # Compare current ATR to moving average
            atr_ratio = self.atr[0] / self.atr_ma[0]
            # Higher volatility -> lower scale factor
            vol_scaler = max(
                self.p.atr_scale_min,
                min(self.p.atr_scale_max, 1.0 / atr_ratio)
            )
            
        # Combined factor (could weight differently if desired)
        combined_factor = vol_flow_factor * vol_scaler
        
        # Store signal
        self.signals.append({
            'bar_index': len(self.data5m) - 1,
            'datetime': dt_utc,
            'obv': self.obv[0],
            'obv_change': obv_change if len(self.obv) > self.p.obv_lookback else 0,
            'vol_flow_factor': vol_flow_factor,
            'atr': self.atr[0],
            'atr_ma': self.atr_ma[0],
            'vol_scaler': vol_scaler,
            'combined_factor': combined_factor,
            'close': self.data5m.close[0],
        })

    def stop(self):
        """Analyze predictive power of volume/volatility factors."""
        if not self.signals:
            print("No signals generated!")
            return

        df = pd.DataFrame(self.signals)
        close_array = np.array(self.data5m.close.array)

        print("\n=== Volume & Volatility Analysis (5m Data) ===")
        print(f"Total 5m Bars: {len(self.data5m)}")
        print(f"Signal Rows: {len(df)}")
        print(f"Signal Rate: {len(df)/len(self.data5m)*100:.1f}% of 5m bars")

        # Factor distribution
        print("\nFactor Distribution:")
        print(f"Volume Flow Factor: {df['vol_flow_factor'].value_counts().to_dict()}")
        print(f"Volatility Scaler Range: [{df['vol_scaler'].min():.2f}, {df['vol_scaler'].max():.2f}]")
        print(f"Combined Factor Range: [{df['combined_factor'].min():.2f}, {df['combined_factor'].max():.2f}]")

        # Analyze forward returns
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

            # Correlate factors with forward returns
            for factor in ['vol_flow_factor', 'vol_scaler', 'combined_factor']:
                ic = valid[factor].corr(valid[col_name])
                n = len(valid)
                if abs(ic) < 1.0:
                    t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
                else:
                    t_stat = np.inf
                    p_val = 0.0

                print(f"\n--- {period}-Bar Forward Returns vs {factor} ---")
                print(f"Number of Valid Signals: {n}")
                print(f"Information Coefficient (IC): {ic:.3f}, p-value: {p_val:.3f}")

                # Average returns by factor quartiles
                quartiles = pd.qcut(valid[factor], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                avg_by_quartile = valid.groupby(quartiles)[col_name].mean() * 100
                print("\nAverage Returns by Factor Quartile:")
                for q, avg_ret in avg_by_quartile.items():
                    print(f"{q}: {avg_ret:.3f}%")


def run_vol_analysis(csv_path):
    """
    Run volume & volatility analysis:
    1. Load 1-min data
    2. Resample to 5-min
    3. Generate and analyze factors
    """
    cerebro = bt.Cerebro()

    # Load 1-min data
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
    cerebro.adddata(data1m, name='1m')

    # Resample to 5-min
    data5m = cerebro.resampledata(
        data1m,
        timeframe=bt.TimeFrame.Minutes,
        compression=5,
        name='5m'
    )

    # Add strategy and run
    cerebro.addstrategy(VolumeVolatilityStrategy)
    cerebro.run()


if __name__ == '__main__':
    """Test with AAPL."""
    csv_file = 'data/historical_data/AAPL_1m_20231227_20241226.csv'
    run_vol_analysis(csv_file)
