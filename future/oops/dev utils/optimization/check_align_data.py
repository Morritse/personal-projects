import pandas as pd
import os
from datetime import datetime

def load_and_check_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def main():
    data_dir = "data"
    data_ranges = {}
    
    # Load all data files and check their date ranges
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and not filename.startswith("6"):
            filepath = os.path.join(data_dir, filename)
            symbol = filename.split('_')[0]
            
            df = load_and_check_data(filepath)
            data_ranges[symbol] = {
                'start': df['Date'].min(),
                'end': df['Date'].max(),
                'rows': len(df)
            }
    
    # Find common date range
    common_start = max(info['start'] for info in data_ranges.values())
    common_end = min(info['end'] for info in data_ranges.values())
    
    print("\nData ranges for each symbol:")
    print("Symbol | Start Date | End Date | Rows")
    print("-" * 50)
    for symbol, info in sorted(data_ranges.items()):
        print(f"{symbol:6} | {info['start'].date()} | {info['end'].date()} | {info['rows']}")
    
    print("\nCommon date range that would work for all symbols:")
    print(f"Start: {common_start.date()}")
    print(f"End: {common_end.date()}")
    
    # Align all data files to common range
    print("\nAligning data files to common range...")
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and not filename.startswith("6"):
            filepath = os.path.join(data_dir, filename)
            symbol = filename.split('_')[0]
            
            df = load_and_check_data(filepath)
            mask = (df['Date'] >= common_start) & (df['Date'] <= common_end)
            df_aligned = df.loc[mask]
            
            # Save aligned data
            df_aligned.to_csv(filepath, index=False)
            print(f"Aligned {symbol}: {len(df_aligned)} rows")

if __name__ == "__main__":
    main()
