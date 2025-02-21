import pandas as pd
from datetime import datetime
import pytz

def copy_cache_data():
    """Create market_data.pkl from historical CSV files."""
    # Load CSV data
    jnj_data = pd.read_csv('data/historical/jnj_hourly.csv', index_col=0, parse_dates=True)
    xlv_data = pd.read_csv('data/historical/xlv_hourly.csv', index_col=0, parse_dates=True)
    
    # Filter to 2022-2024 (optimization period)
    start_date = datetime(2021, 1, 1, tzinfo=pytz.UTC)
    end_date = datetime(2024, 12, 31, tzinfo=pytz.UTC)
    
    jnj_data = jnj_data[(jnj_data.index >= start_date) & (jnj_data.index <= end_date)]
    xlv_data = xlv_data[(xlv_data.index >= start_date) & (xlv_data.index <= end_date)]
    
    # Remove timezone info
    jnj_data.index = jnj_data.index.tz_localize(None)
    xlv_data.index = xlv_data.index.tz_localize(None)
    
    # Create market data dict
    market_data = {
        'JNJ': jnj_data,
        'XLV': xlv_data
    }
    
    # Save to market_data.pkl
    pd.to_pickle(market_data, 'data/market_data.pkl')
    
    # Print summary
    print("\nData saved to market_data.pkl:")
    for symbol, df in market_data.items():
        print(f"\n{symbol}:")
        print(f"Date Range: {df.index[0]} to {df.index[-1]}")
        print(f"Total Hours: {len(df)}")
        print("\nTrading Hours Distribution:")
        print(df.index.hour.value_counts().sort_index())

if __name__ == "__main__":
    copy_cache_data()
