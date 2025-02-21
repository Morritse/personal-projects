import pandas as pd
import numpy as np

def check_data():
    """Check what's happening with the data."""
    # Load data
    data = pd.read_pickle('data/market_data.pkl')
    jnj_df = data['JNJ']
    xlv_df = data['XLV']
    
    # Check each year
    years = [2021, 2022, 2023, 2024]
    
    for year in years:
        start_date = pd.Timestamp(f"{year}-01-01")
        end_date = pd.Timestamp(f"{year}-12-31")
        
        jnj_year = jnj_df[(jnj_df.index >= start_date) & (jnj_df.index <= end_date)]
        xlv_year = xlv_df[(xlv_df.index >= start_date) & (xlv_df.index <= end_date)]
        
        print(f"\nYear {year}:")
        print("-" * 50)
        
        print("\nJNJ:")
        print(f"Date Range: {jnj_year.index[0]} to {jnj_year.index[-1]}")
        print(f"Total Hours: {len(jnj_year)}")
        print("\nTrading Hours Distribution:")
        print(jnj_year.index.hour.value_counts().sort_index())
        
        print("\nXLV:")
        print(f"Date Range: {xlv_year.index[0]} to {xlv_year.index[-1]}")
        print(f"Total Hours: {len(xlv_year)}")
        print("\nTrading Hours Distribution:")
        print(xlv_year.index.hour.value_counts().sort_index())
        
        # Check returns
        print("\nReturns Check:")
        returns = xlv_year['close'].pct_change()
        ret = returns.rolling(window=20).mean() * 252
        vol = returns.rolling(window=20).std() * np.sqrt(252)
        
        print("\nReturns stats:")
        print(ret.describe())
        print("\nVolatility stats:")
        print(vol.describe())
        
        # Check for gaps
        print("\nChecking for gaps:")
        time_diff = xlv_year.index.to_series().diff()
        gaps = time_diff[time_diff > pd.Timedelta(hours=1)]
        if not gaps.empty:
            print(f"Found {len(gaps)} gaps:")
            for idx, gap in gaps.head().items():
                print(f"Gap at {idx}: {gap}")

if __name__ == "__main__":
    check_data()
