import os
import json
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from dotenv import load_dotenv
from config_bt import SYMBOLS

# Load Alpaca credentials
load_dotenv()
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')


def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    cache_dir = 'cache_data'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def download_crypto_data(symbol, start_date, end_date, interval='1m'):
    """Download historical crypto data using Alpaca Crypto API"""
    try:
        # Convert dates to RFC-3339 format
        start_rfc = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_rfc = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Map interval to Alpaca's timeframe
        timeframe_map = {
            '1m': tradeapi.TimeFrame.Minute,
            '5m': tradeapi.TimeFrame(5, tradeapi.TimeFrameUnit.Minute),
            '15m': tradeapi.TimeFrame(15, tradeapi.TimeFrameUnit.Minute),
            '1h': tradeapi.TimeFrame.Hour,
            '1d': tradeapi.TimeFrame.Day
        }
        timeframe = timeframe_map.get(interval, tradeapi.TimeFrame.Minute)
        
        # Get crypto bars
        bars = api.get_crypto_bars(
            symbol,
            timeframe,
            start=start_rfc,
            end=end_rfc
        ).df
        
        if bars.empty:
            print(f"No data returned for {symbol}")
            return None
            
        # Rename columns to match existing format
        df = bars.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        return df
        
    except Exception as e:
        print(f"Error downloading crypto data for {symbol}: {e}")
        return None

def download_data(start_date, end_date, interval='1m'):
    """Download historical data for all symbols"""
    cache_dir = ensure_cache_dir()
    data = {}
    metadata = {
        'last_updated': datetime.now().isoformat(),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'interval': interval,
        'symbols': {}
    }
    
    # Convert dates to strings
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    for symbol in SYMBOLS:
        print(f"Downloading {symbol}...")
        
        try:
            if symbol.upper() == 'VIX':
                # Use Yahoo Finance for VIX
                df = yf.download('^VIX', start=start_str, end=end_str, interval=interval)
                if df.empty:
                    print(f"No data returned for {symbol}")
                    continue
                    
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
            elif symbol in SYMBOLS:
                # Use Alpaca Crypto API for crypto symbols
                df = download_crypto_data(symbol, start_date, end_date, interval)
                if df is None:
                    continue
            else:
                # Use Alpaca for stocks
                start_rfc = f"{start_str}T00:00:00Z"
                end_rfc = f"{end_str}T23:59:59Z"
                
                bars = api.get_bars(
                    symbol,
                    tradeapi.TimeFrame.Hour,
                    start=start_rfc,
                    end=end_rfc
                ).df
                
                if bars.empty:
                    print(f"No data returned for {symbol}")
                    continue
                
                if isinstance(bars.index, pd.MultiIndex):
                    bars = bars.xs(symbol, level=0)
                
                df = bars.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
            
            # Save to cache - replace slashes with underscores for crypto symbols
            cache_symbol = symbol.lower().replace('/', '_')
            cache_file = os.path.join(cache_dir, f"{cache_symbol}_data.csv")
            # Ensure the cache directory exists
            ensure_cache_dir()
            df.to_csv(cache_file)
            
            # Update metadata
            metadata['symbols'][symbol] = {
                'rows': len(df),
                'first_date': df.index[0].isoformat() if len(df) > 0 else None,
                'last_date': df.index[-1].isoformat() if len(df) > 0 else None,
                'cache_file': cache_file
            }
            
            data[symbol] = df
            print(f"Successfully downloaded {symbol} ({len(df)} rows)")
            
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            continue
    
    # Save metadata
    with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return data, metadata

def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample data to a higher timeframe"""
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    return resampled.dropna()

def load_cached_data(symbol, timeframe='1m'):
    """Load data for a symbol from cache"""
    cache_dir = ensure_cache_dir()
    cache_file = os.path.join(cache_dir, f"{symbol.lower()}_data.csv")
    
    if os.path.exists(cache_file):
        df = pd.read_csv(
            cache_file,
            index_col=0,
            parse_dates=True,
            dtype={
                'open': float,
                'high': float,
                'low': float,
                'close': float,
                'volume': float
            }
        )
        # If requesting a higher timeframe, resample the data
        if timeframe != '1m':
            df = resample_to_timeframe(df, timeframe)
        return df
    return None

if __name__ == '__main__':
    # Download two years of hourly data
    # Download 30 days of minute data (to avoid hitting API limits)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    
    print(f"Downloading data from {start_date.date()} to {end_date.date()}...")
    data, metadata = download_data(start_date, end_date)
    
    print("\nDownload Summary:")
    for symbol, info in metadata['symbols'].items():
        print(f"{symbol}: {info['rows']} rows from {info['first_date']} to {info['last_date']}")
