import os
import time
from datetime import datetime, timedelta
import pytz

import pandas as pd
import numpy as np
from alpaca_trade_api.rest import REST, TimeFrame
import pandas_ta as ta  # Added pandas-ta for technical indicators
import yfinance as yf  # Added for VIX data

# Alpaca API configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_SECRET_KEY', 'YOUR_API_SECRET')
BASE_URL = "https://paper-api.alpaca.markets"  # Use paper trading for testing

# Default data directory paths.  Adjust as needed.
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

def get_alpaca_client():
    """Initialize Alpaca API client."""
    return REST(
        key_id=ALPACA_API_KEY,
        secret_key=ALPACA_API_SECRET,
        base_url=BASE_URL,
        api_version='v2'
    )

def fetch_daily_data(symbol, start_date, end_date):
    """
    Fetch daily data from Alpaca or Yahoo Finance.

    Parameters:
    - symbol: Stock symbol (e.g., 'AAPL')
    - start_date: Start date for historical data (e.g., '2015-01-01')
    - end_date: End date for historical data (e.g., '2023-10-01')
    """
    print(f"\nFetching daily data for {symbol} from {start_date} to {end_date}...")

    try:
        if symbol == '^VIX':  # Use Yahoo Finance for VIX data
            ticker = yf.Ticker(symbol)
            bars = ticker.history(start=start_date, end=end_date, interval='1d')
            bars = bars.reset_index()
            bars.rename(columns={
                'Date': 'datetime',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
        else:  # Use Alpaca for stocks and ETFs
            api = get_alpaca_client()
            tf = TimeFrame.Day
            bars = api.get_bars(
                symbol,
                tf,
                start=start_date,
                end=end_date,
                adjustment='all'
            ).df
            bars = bars.reset_index()
            bars.rename(columns={'timestamp': 'datetime'}, inplace=True)

        # Add symbol column
        bars['symbol'] = symbol

        # Save raw data
        filename = f"{symbol}_raw_daily.csv"
        os.makedirs(RAW_DATA_PATH, exist_ok=True)
        bars.to_csv(os.path.join(RAW_DATA_PATH, filename), index=False)

        print(f"Raw data saved to {filename}")
        print(f"Time range: {bars['datetime'].min()} to {bars['datetime'].max()}")
        print(f"Number of bars: {len(bars)}")

        return bars

    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(data):
    """Calculate technical indicators using pandas-ta."""
    # Create copy of data
    df = data.copy()
    
    try:
        # Calculate Moving Averages
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        df['ema_10'] = ta.ema(df['close'], length=10)
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['ema_200'] = ta.ema(df['close'], length=200)
        
        # Calculate RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # Calculate MACD
        macd = ta.macd(df['close'])
        df['macd'] = macd.iloc[:, 0]  # MACD line
        df['macd_signal'] = macd.iloc[:, 1]  # Signal line
        df['macd_diff'] = macd.iloc[:, 2]  # MACD histogram
        
        # Calculate Stochastic
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.iloc[:, 0]  # %K line
        df['stoch_d'] = stoch.iloc[:, 1]  # %D line
        
        # Calculate ADX
        adx = ta.adx(df['high'], df['low'], df['close'])
        df['adx'] = adx.iloc[:, 0]  # ADX
        df['di_plus'] = adx.iloc[:, 1]  # DI+
        df['di_minus'] = adx.iloc[:, 2]  # DI-
        
        # Calculate Bollinger Bands
        bb = ta.bbands(df['close'])
        df['bb_lower'] = bb.iloc[:, 0]  # Lower band
        df['bb_middle'] = bb.iloc[:, 1]  # Middle band
        df['bb_upper'] = bb.iloc[:, 2]  # Upper band
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate ATR
        df['atr'] = ta.atr(df['high'], df['low'], df['close'])
        
        # Calculate Volume Indicators
        df['obv'] = ta.obv(df['close'], df['volume'])
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        
        # Additional Calculations
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        df['returns_volatility'] = df['returns'].rolling(window=20).std()
        df['volume_ma_20'] = ta.sma(df['volume'], length=20)
        df['volume_ma_50'] = ta.sma(df['volume'], length=50)
        
        # Price relative to moving averages
        df['price_to_sma_200'] = df['close'] / df['sma_200'] - 1
        df['price_to_sma_50'] = df['close'] / df['sma_50'] - 1
        
        # Moving average crossovers
        df['sma_50_200_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, -1)
        df['ema_50_200_cross'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)
        
        # Remove rows with NaN values
        df.dropna(inplace=True)

        # Debugging: Print the columns to verify
        print("Calculated columns:", df.columns.tolist())
        
        return df
    
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        raise

def process_daily_data(df, symbol):
    """
    Process daily data for the model.

    Parameters:
    - df: Raw DataFrame from fetch_daily_data
    - symbol: Stock symbol
    """
    if df.empty:
        print(f"No data to process for {symbol}")
        return df

    print(f"\nProcessing daily data for {symbol}...")

    # Create copy to avoid modifying original
    data = df.copy()

    # Ensure datetime is timezone-aware and set to UTC
    data['datetime'] = pd.to_datetime(data['datetime'], utc=True)

    # Sort by datetime
    data.sort_values('datetime', inplace=True)

    # Keep only necessary columns
    data = data[['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

    # Calculate technical indicators
    data = calculate_technical_indicators(data)

    # Debugging: Print the processed data columns
    print("Processed data columns:", data.columns.tolist())

    # Save processed data
    filename = f"{symbol}_processed_daily.csv"
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    data.to_csv(os.path.join(PROCESSED_DATA_PATH, filename), index=False)

    print(f"Processed data saved to {filename}")
    print(f"Final shape: {data.shape}")

    return data

def prepare_daily_dataset(symbol, start_date=None, end_date=None):
    """
    Prepare complete daily dataset for training.

    Parameters:
    - symbol: Stock symbol
    - start_date: Start date for historical data (default: 5 years ago)
    - end_date: End date for historical data (default: today)
    """
    if end_date is None:
        end_date = datetime.now(pytz.UTC).strftime('%Y-%m-%d')

    if start_date is None:
        # Default to 5 years of historical data
        start_date = (datetime.now(pytz.UTC) - timedelta(days=5*365)).strftime('%Y-%m-%d')

    # Handle VIX symbol
    if symbol == 'VIX':
        symbol = '^VIX'  # Yahoo Finance format

    # Fetch raw data
    raw_data = fetch_daily_data(symbol, start_date, end_date)

    if raw_data.empty:
        print(f"No data available for {symbol}")
        return pd.DataFrame()

    # Process data
    processed_data = process_daily_data(raw_data, symbol.replace('^', ''))  # Remove ^ for file naming

    return processed_data

def update_daily_data(symbols):
    """
    Update daily data for multiple symbols with error handling and rate limiting.

    Parameters:
    - symbols: List of stock symbols
    """
    end_date = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
    start_date = (datetime.now(pytz.UTC) - timedelta(days=5*365)).strftime('%Y-%m-%d')

    for symbol in symbols:
        retries = 3
        while retries > 0:
            try:
                prepare_daily_dataset(symbol, start_date, end_date)
                print(f"\nSuccessfully updated data for {symbol}")
                time.sleep(1)  # Sleep for 1 second between symbols
                break
            except Exception as e:
                print(f"\nError updating data for {symbol}: {str(e)}")
                retries -= 1
                if retries == 0:
                    print(f"Failed to update data for {symbol} after multiple attempts.")
                else:
                    print(f"Retrying... ({3 - retries} of 3)")

def extract_features(data):
    """
    Extract relevant features for model training.

    Parameters:
    - data: DataFrame containing processed stock data

    Returns:
    - features: DataFrame containing the features for model training
    """
    features = data[['returns', 'rsi', 'volume', 'returns_volatility']].copy()
    features['target'] = np.where(features['returns'] > 0, 1, 0)  # Binary target for classification
    return features

def create_sliding_windows(data, window_size, step_size):
    """
    Create sliding windows from the data for model input.

    Parameters:
    - data: Input data (e.g., OHLCV)
    - window_size: Size of the sliding window
    - step_size: Step size for sliding the window

    Returns:
    - np.array of sliding windows
    """
    sequences = []
    for i in range(0, len(data) - window_size + 1, step_size):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)
