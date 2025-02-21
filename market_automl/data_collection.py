import alpaca_trade_api as tradeapi
import pandas as pd
from datetime import datetime, timedelta
from config import (
    ALPACA_API_KEY, 
    ALPACA_SECRET_KEY, 
    ALPACA_BASE_URL,
    TECH_STOCKS,
    INDEX_FUNDS,
    TIMEFRAME,
    LOOKBACK_PERIOD
)

class DataCollector:
    def __init__(self):
        self.api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            ALPACA_BASE_URL,
            api_version='v2'
        )
        
    def get_historical_data(self, symbol, start_date=None, end_date=None):
        """Fetch historical data for a given symbol."""
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - pd.Timedelta(LOOKBACK_PERIOD)
            
        df = self.api.get_bars(
            symbol,
            timeframe=TIMEFRAME,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            adjustment='raw'
        ).df
        
        return df
    
    def fetch_all_data(self):
        """Fetch data for all configured symbols."""
        all_symbols = TECH_STOCKS + INDEX_FUNDS
        data_dict = {}
        
        for symbol in all_symbols:
            try:
                print(f"Fetching data for {symbol}")
                data_dict[symbol] = self.get_historical_data(symbol)
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")
                
        return data_dict

    def save_data(self, data_dict, base_path='data/raw'):
        """Save collected data to CSV files."""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        for symbol, df in data_dict.items():
            filepath = f"{base_path}/{symbol}_raw.csv"
            df.to_csv(filepath)
            print(f"Saved data for {symbol} to {filepath}")

def main():
    collector = DataCollector()
    data = collector.fetch_all_data()
    collector.save_data(data)

if __name__ == "__main__":
    main()
