import os
import pandas as pd

def clean_live_data():
    live_data_dir = 'live_data'
    
    for filename in os.listdir(live_data_dir):
        if not filename.endswith('.csv'):
            continue
            
        symbol = filename.replace('.csv', '')
        filepath = os.path.join(live_data_dir, filename)
        print(f"Processing {filename}...")
        
        # Read the CSV
        df = pd.read_csv(filepath)
        
        # Print current columns for debugging
        print(f"Current columns: {df.columns.tolist()}")
        
        # Rename columns to lowercase if needed
        rename_map = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Ensure all required columns exist
        if 'average' not in df.columns:
            df['average'] = 0.0
        if 'barCount' not in df.columns:
            df['barCount'] = 0.0
        if 'localSymbol' not in df.columns:
            df['localSymbol'] = f"{symbol}=F"
        if 'expiry' not in df.columns:
            df['expiry'] = ''
            
        # Ensure columns are in correct order
        columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'average', 'barCount', 'localSymbol', 'expiry']
        df = df[columns]
        
        # Save back to file
        df.to_csv(filepath, index=False)
        print(f"Cleaned {filename}")

if __name__ == "__main__":
    clean_live_data()
