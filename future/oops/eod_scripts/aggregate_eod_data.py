import pandas as pd
import os

def clean_date(date_str):
    """Convert any date string to YYYY-MM-DD format without timezone."""
    try:
        return pd.to_datetime(date_str).strftime('%Y-%m-%d')
    except:
        return None

def aggregate_eod_data():
    # Read the EOD snapshot
    snapshot_file = 'daily_data/futures_eod_snapshot.csv'
    new_data = pd.read_csv(snapshot_file)
    new_data['date'] = new_data['date'].apply(clean_date)
    
    print(f"\nProcessing new data from {new_data['date'].iloc[0]}")
    
    # Process each symbol in the new data
    for symbol in new_data['symbol'].unique():
        # Get new data for this symbol
        symbol_new_data = new_data[new_data['symbol'] == symbol].copy()
        new_date = symbol_new_data['date'].iloc[0]
        
        # Path to live data file
        live_file = f'live_data/{symbol}.csv'
        
        if os.path.exists(live_file):
            # Read existing live data
            live_df = pd.read_csv(live_file)
            live_df['date'] = live_df['date'].apply(clean_date)
            
            # Check if we already have this date
            if new_date in live_df['date'].values:
                print(f"{symbol}: Already have data for {new_date}")
                continue
            
            # Combine with new data
            combined_df = pd.concat([live_df, symbol_new_data.drop('symbol', axis=1)])
            
            # Remove duplicates, keeping latest version of any duplicate dates
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            
            # Sort by date
            combined_df.sort_values('date', inplace=True)
            
            # Save back to file
            combined_df.to_csv(live_file, index=False)
            print(f"{symbol}: Added {len(symbol_new_data)} new row(s) for {new_date}")
        else:
            print(f"Warning: No live data file found for {symbol}")

if __name__ == "__main__":
    aggregate_eod_data()
