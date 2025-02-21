# data_manager.py
import os
import pandas as pd

class DataManager:
    def __init__(self, data_folder='historical_data'):
        """
        data_folder is where you store CSV files like:
           AAPL.csv, MSFT.csv, etc.
        Each CSV might have columns:
           timestamp,open,high,low,close,volume,...
        """
        self.data_folder = data_folder

    def load_historical_data(self, symbol):
        """
        Attempt to load CSV data for a given symbol.
        The CSV file is expected to be located at:
           <self.data_folder>/<symbol>.csv
        Returns a DataFrame with a DateTimeIndex if successful,
        else None.
        """
        filename = f"{symbol}.csv"
        filepath = os.path.join(self.data_folder, filename)
        
        if not os.path.isfile(filepath):
            print(f"File not found for {symbol}: {filepath}")
            return None
        
        try:
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            
            # Set index to timestamp
            df.set_index('timestamp', inplace=True)
            
            # If your CSV columns are named differently, rename them:
            # e.g. your CSV has 'open', 'high', 'low', 'close' => no rename needed
            # but if your CSV has 'close,high,low,open,...' we fix that:
            rename_map = {}
            if 'open' not in df.columns and 'Open' in df.columns:
                rename_map['Open'] = 'open'
            if 'close' not in df.columns and 'close' in df.columns:
                pass  # already correct
            if 'high' not in df.columns and 'High' in df.columns:
                rename_map['High'] = 'high'
            if 'low' not in df.columns and 'Low' in df.columns:
                rename_map['Low'] = 'low'
            
            # If your CSV has columns with different names (like 'close' -> 'close' is fine,
            # but if you have something like 'ClosePrice' => rename_map['ClosePrice'] = 'close')
            # Example for your CSV with columns: timestamp,close,high,low,open,volume
            # rename_map = {
            #     'close': 'close',
            #     'high': 'high',
            #     'low': 'low',
            #     'open': 'open',
            # }
            
            # For the snippet you provided:
            #  timestamp,close,high,low,trade_count,open,volume,vwap
            # we want to rename them to open, high, low, close
            # so:
            rename_map.update({
                'close': 'close',
                'high': 'high',
                'low': 'low',
                'open': 'open'
            })
            
            df.rename(columns=rename_map, inplace=True)
            
            # Sort by index just to be safe
            df.sort_index(inplace=True)
            
            return df
        
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
