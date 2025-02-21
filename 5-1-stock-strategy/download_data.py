import os
import json
import pandas as pd
import sys
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone
import pytz
from dotenv import load_dotenv

# Load Alpaca credentials
load_dotenv()
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')

# Use NY timezone for market hours
NY_TZ = pytz.timezone('America/New_York')

SYMBOLS = [
    # High Growth Tech
    'META',
    'COIN',
    'SQ',
    'ROKU',
    'PYPL',
    'CRWD',
    'SNOW',
    'BNTX',
    'AAPL',
    'AMZN',
    'NFLX'
]

def ensure_cache_dir():
    """Create cache directory if it doesn't exist"""
    cache_dir = 'cache_data'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

def download_data(start_date, end_date, interval='1Min', force_update=False):
    """Download historical data for all symbols"""
    cache_dir = ensure_cache_dir()
    data = {}
    
    # Load existing metadata if available
    metadata_file = os.path.join(cache_dir, 'metadata.json')
    cached_symbols = set()
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            last_updated = datetime.fromisoformat(metadata['last_updated'])
            # If data is less than a day old and not forcing update
            if not force_update and (datetime.now(NY_TZ) - last_updated.astimezone(NY_TZ)).days < 1:
                print("\nChecking cached data (less than 24 hours old):")
                for symbol in SYMBOLS:
                    df = load_cached_data(symbol)
                    if df is not None:
                        cached_symbols.add(symbol)
                        data[symbol] = df
                
                # Show what we have and what we need
                print("\nAlready cached:")
                for symbol in sorted(cached_symbols):
                    print(f"- {symbol}")
                
                new_symbols = set(SYMBOLS) - cached_symbols
                if new_symbols:
                    print("\nNeed to download:")
                    for symbol in sorted(new_symbols):
                        print(f"- {symbol}")
                    print("\nProceeding with download of new symbols...")
                else:
                    print("\nAll symbols are cached and up to date")
                    return data, metadata
    
    # Initialize new metadata
    metadata = {
        'last_updated': datetime.now(NY_TZ).isoformat(),
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'interval': interval,
        'symbols': {}
    }
    
    # Convert dates to RFC3339 format in UTC (required by Alpaca)
    start_str = start_date.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_str = end_date.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Download only new or forced symbols
    symbols_to_download = SYMBOLS if force_update else (set(SYMBOLS) - cached_symbols)
    for symbol in symbols_to_download:
        print(f"\nDownloading {symbol}...")
        
        try:
            if symbol.upper() == 'VIX':
                # Use Yahoo Finance for VIX
                df = yf.download('^VIX', start=start_date, end=end_date, interval=interval)
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
                # Convert index to NY timezone
                df.index = df.index.tz_localize('UTC').tz_convert(NY_TZ)
            else:
                # Use Alpaca for stocks - fetch data in 5-day chunks
                all_bars = []
                chunk_start = start_date
                chunk_size = timedelta(days=10)
                
                while chunk_start < end_date:
                    chunk_end = min(chunk_start + chunk_size, end_date)
                    chunk_start_str = chunk_start.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                    chunk_end_str = chunk_end.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                    
                    print(f"  Fetching {chunk_start.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}...")
                    
                    bars_response = api.get_bars(
                        symbol,
                        tradeapi.TimeFrame.Minute,
                        start=chunk_start_str,
                        end=chunk_end_str
                    )
                    
                    if not bars_response.df.empty:
                        all_bars.append(bars_response.df)
                    
                    chunk_start = chunk_end
                
                if not all_bars:
                    print(f"No data returned for {symbol}")
                    continue
                
                # Combine all chunks and sort by time
                bars = pd.concat(all_bars).sort_index()
                
                if isinstance(bars.index, pd.MultiIndex):
                    bars = bars.xs(symbol, level=0)
                
                # Convert index to NY timezone
                bars.index = bars.index.tz_convert(NY_TZ)
                
                df = bars.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                })
            
            # Save to cache
            cache_file = os.path.join(cache_dir, f"{symbol.lower()}_data.csv")
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

def load_cached_data(symbol):
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
        # Convert index to NY timezone if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            # If index is not DatetimeIndex, create one without timezone first
            df.index = pd.to_datetime(df.index, utc=True)
        
        # Now handle timezone conversion
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Convert to NY timezone if not already
        if df.index.tz != NY_TZ:
            df.index = df.index.tz_convert(NY_TZ)
        return df
    return None

if __name__ == '__main__':
    # End at current time
    end_date = datetime.now(NY_TZ) - timedelta(days=1)
    # Start 30 days ago at market open
    start_date = (end_date - timedelta(days=200)).replace(hour=9, minute=30, second=0, microsecond=0)
    
    # Add force_update flag
    force_update = False
    if '--force' in sys.argv:
        force_update = True
        print("Force update enabled - will re-download all data")
    
    print(f"Downloading data from {start_date.strftime('%Y-%m-%d %H:%M %Z')} to {end_date.strftime('%Y-%m-%d %H:%M %Z')}...")
    data, metadata = download_data(start_date, end_date, force_update=force_update)
    
    print("\nDownload Summary:")
    for symbol, info in metadata['symbols'].items():
        print(f"{symbol}: {info['rows']} rows from {info['first_date']} to {info['last_date']}")
