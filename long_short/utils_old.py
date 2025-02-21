import os
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta

# Default Alpaca API credentials.  Replace with user-provided credentials.
API_KEY = "PKMNC0G9A58H9C2IVKP2"
API_SECRET = "Z0pq7QurDcxCMcgCO8LAtFdWqHy6RHro6lr6fmxi"
BASE_URL = "https://paper-api.alpaca.markets" # Use paper trading for testing

# Default list of stocks. Replace with user-provided list.
STOCKS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "META", "BRK.B", "JPM", "V"]

def fetch_data(api_key, api_secret, base_url, symbols, timeframe, start_date, end_date):
    """Fetch historical stock data from Alpaca API."""
    try:
        api = tradeapi.REST(api_key, api_secret, base_url, api_version='v2')
        data = api.get_barset(symbols, timeframe, start=start_date, end=end_date, limit=10000).df
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def create_features(data):
    """Create features for the model."""
    try:
        data['return'] = data.groupby('symbol')['close'].pct_change()
        data['volume_change'] = data.groupby('symbol')['volume'].pct_change()
        data = data.dropna()
        return data
    except KeyError as e:
        print(f"Error creating features: {e}. Check if 'close' and 'volume' columns exist in your data.")
        return None

def preprocess_data(data):
    """Preprocess the data for model training."""
    try:
        # Group data by symbol
        grouped = data.groupby('symbol')

        # Create a list to store preprocessed data for each symbol
        preprocessed_data = []

        # Iterate through each symbol
        for symbol, group in grouped:
            # Select relevant columns
            group = group[['return', 'volume_change']]

            # Normalize data (example: min-max scaling)
            group = (group - group.min()) / (group.max() - group.min())

            # Add symbol column back
            group['symbol'] = symbol

            # Append to the list
            preprocessed_data.append(group)

        # Concatenate data for all symbols
        preprocessed_data = pd.concat(preprocessed_data)

        return preprocessed_data
    except KeyError as e:
        print(f"Error preprocessing data: {e}. Check if 'return' and 'volume_change' columns exist.")
        return None

def prepare_data(api_key, api_secret, base_url, symbols, timeframe, start_date, end_date):
    """Fetch, create features, and preprocess data."""
    raw_data = fetch_data(api_key, api_secret, base_url, symbols, timeframe, start_date, end_date)
    if raw_data is None:
        return None
    featured_data = create_features(raw_data)
    if featured_data is None:
        return None
    preprocessed_data = preprocess_data(featured_data)
    return preprocessed_data

# Example usage:
start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()
data = prepare_data(API_KEY, API_SECRET, BASE_URL, STOCKS, 'minute', start_date, end_date)
if data is not None:
    print(data.head())
