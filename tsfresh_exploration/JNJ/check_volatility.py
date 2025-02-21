import pandas as pd
import numpy as np
from datetime import datetime

def check_volatility():
    """Check what threshold we're using in the original version."""
    # Load data
    data = pd.read_pickle('data/market_data.pkl')
    xlv_data = data['XLV']
    
    # Filter to 2022-2024
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    xlv_data = xlv_data[(xlv_data.index >= start_date) & (xlv_data.index <= end_date)]
    
    # Calculate returns and rolling volatility
    returns = xlv_data['close'].pct_change()
    vol = returns.rolling(window=20).std() * np.sqrt(252)
    
    # Calculate threshold using original method
    threshold = vol.quantile(0.67)
    print(f"\nOriginal threshold (67th percentile): {threshold:.6f}")
    
    # Print distribution
    print("\nVolatility distribution:")
    print(vol.describe())
    
    # Print % of time above threshold
    time_above = (vol > threshold).mean() * 100
    print(f"\nTime spent above threshold: {time_above:.1f}%")
    
    # Compare to VIX levels
    print("\nTypical market volatility levels:")
    print("Low volatility:    < 15%")
    print("Normal volatility: 15-25%")
    print("High volatility:   > 25%")
    
    # Print highest vol periods
    print("\nHighest volatility periods:")
    high_vol = vol[vol > 0.047].sort_values(ascending=False)
    for date, v in high_vol.head().items():
        print(f"{date.strftime('%Y-%m-%d %H:%M')}: {v:.1%}")

if __name__ == "__main__":
    check_volatility()
