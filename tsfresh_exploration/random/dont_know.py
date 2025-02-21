import os
import time
from datetime import datetime, timedelta
import pandas as pd
import alpaca_trade_api as tradeapi
from get_random_symbol import get_random_symbol
import deep_analysis
from top_symbols import TopSymbolTracker

# Initialize top symbols tracker
tracker = TopSymbolTracker()

###############################################
# 1) SETUP ALPACA CONNECTION
###############################################
# Load and verify environment variables
from dotenv import load_dotenv
print("Loading environment variables...")
load_dotenv(override=True)  # Force reload of environment variables

API_KEY_ID = os.getenv('ALPACA_API_KEY')
API_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

print("Environment variables loaded:")
print(f"API Key found: {'Yes' if API_KEY_ID else 'No'}")
print(f"Secret Key found: {'Yes' if API_SECRET_KEY else 'No'}")
BASE_URL = 'https://paper-api.alpaca.markets'  # Explicitly set for paper trading

# Verify API credentials
if not API_KEY_ID or not API_SECRET_KEY:
    raise ValueError("API credentials not found in environment variables")

print(f"Connecting to Alpaca API at {BASE_URL}")
print(f"Using API key: {API_KEY_ID[:5]}...")  # Show first 5 chars for verification

api = tradeapi.REST(
    API_KEY_ID,
    API_SECRET_KEY,
    base_url=BASE_URL,
    api_version='v2'
)

# Verify connection
try:
    account = api.get_account()
    print(f"Successfully connected to Alpaca API (Account ID: {account.id})")
except Exception as e:
    print(f"Error connecting to Alpaca API: {str(e)}")
    raise





###############################################
# 3) FETCH 1 YEAR HISTORICAL DAILY DATA
###############################################
def fetch_symbol_data(symbol: str, days=1460) -> pd.DataFrame:
    """
    Fetch hourly data for pattern analysis.
    Default to 4 years of data for longer-term patterns.
    Handles younger symbols gracefully.
    """
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)

    print(f"Fetching {days} days of hourly data for {symbol}...")
    
    try:
        # Try to get asset info first
        asset = api.get_asset(symbol)
        if not asset.tradable:
            print(f"{symbol} is not currently tradable")
            return pd.DataFrame()
            
        # Use hourly data for better signal/noise ratio
        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Hour,
            start=start_dt.strftime('%Y-%m-%d'),
            end=end_dt.strftime('%Y-%m-%d'),
            adjustment='raw'  # Get raw data without adjustments
        )
        
        if len(bars) == 0:
            print(f"No data available for {symbol}")
            return pd.DataFrame()
            
        print(f"Got {len(bars)} hours of data")
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()
    
    df = bars.df
    if df.empty:
        return pd.DataFrame()  # No data
    # If multi-index, flatten
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level=[0,1])
    else:
        df = df.reset_index()

    # Convert to a simpler form
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("America/New_York")
    df = df.sort_values("timestamp")
    # Keep relevant columns
    df = df[["timestamp","open","high","low","close","volume"]]
    df.set_index("timestamp", inplace=True)
    
    return df

def validate_data(df: pd.DataFrame, symbol: str) -> bool:
    """
    Enhanced data validation with minimum requirements.
    At least 1 year of data (≈1,560 hourly points) for meaningful analysis.
    """
    if df.empty:
        print(f"{symbol}: No data returned.")
        return False
        
    min_points = 1560  # Approximately 1 year of hourly data
    if len(df) < min_points:
        print(f"{symbol}: Not enough history. Need {min_points} points, got {len(df)}.")
        return False
        
    # Check for data quality
    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"{symbol}: Found missing values:")
        for col in null_counts.index:
            if null_counts[col] > 0:
                print(f"  - {col}: {null_counts[col]} missing points")
        
        # Only drop if missing data is minimal (< 1%)
        if null_counts.max() / len(df) > 0.01:
            print(f"{symbol}: Too many missing values (>1% of data)")
            return False
            
        print("Dropping rows with missing values...")
        df.dropna(inplace=True)
        
        # Recheck length after dropping
        if len(df) < min_points:
            print(f"{symbol}: Not enough clean data points after dropping nulls.")
            return False
    
    return True

def log_standout_patterns(outcome: dict, min_score: float = 5.0):
    """Log patterns that exceed a minimum interestingness score."""
    if outcome['pattern_score'] >= min_score:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open('standout_patterns.log', 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Time: {timestamp}\n")
            f.write(f"Symbol: {outcome['symbol']}\n")
            f.write(f"Pattern Score: {outcome['pattern_score']}\n")
            f.write(f"Data Points: {outcome['row_count']}\n")
            
            if outcome['major_patterns']:
                f.write("\nMajor Patterns:\n")
                for pattern in outcome['major_patterns']:
                    f.write(f"- {pattern}\n")
            
            if outcome['regime_changes']:
                f.write("\nRegime Changes:\n")
                for change in outcome['regime_changes']:
                    f.write(f"- {change}\n")
            
            if outcome['correlations']:
                f.write("\nCorrelations:\n")
                for corr in outcome['correlations']:
                    f.write(f"- {corr}\n")
            
            if outcome['notes']:
                f.write("\nDetailed Notes:\n")
                for note in outcome['notes']:
                    f.write(f"- {note}\n")
            
            f.write(f"{'='*80}\n")

def run_random_symbol_analysis(num_symbols=3, max_retries=10):
    """Run deep pattern analysis on random symbols."""
    results = []
    standout_threshold = 5.0  # Log patterns with scores >= 5
    
    symbols_analyzed = 0
    attempts = 0
    
    while symbols_analyzed < num_symbols and attempts < num_symbols * max_retries:
        attempts += 1
        sym = get_random_symbol(api)
        print(f"\n=== Analyzing symbol: {sym} ===")
        
        # 1) fetch data
        df = fetch_symbol_data(sym)
        if not validate_data(df, sym):
            print(f"Skipping {sym}, trying another symbol...\n")
            continue
            
        try:
            
            # 2) Run deep analysis
            features, metrics = deep_analysis.extract_time_series_features(df, sym)
            patterns = deep_analysis.detect_patterns(features, metrics)
            
            # 3) Store results
            outcome = {
                "symbol": sym,
                "row_count": len(df),
                "pattern_score": patterns["score"],
                "major_patterns": patterns["major_patterns"],
                "regime_changes": patterns["regime_changes"],
                "correlations": patterns["correlations"],
                "notes": patterns["notes"]
            }
            results.append(outcome)
            
            # 4) Print detailed results
            print(f"\nResults for {sym}:")
            print(f"Data points: {outcome['row_count']}")
            print(f"Pattern Score: {outcome['pattern_score']}")
            
            if outcome['major_patterns']:
                print("\nMajor Patterns Detected:")
                for pattern in outcome['major_patterns']:
                    print(f"- {pattern}")
                    
            if outcome['regime_changes']:
                print("\nRegime Changes:")
                for change in outcome['regime_changes']:
                    print(f"- {change}")
                    
            if outcome['correlations']:
                print("\nCorrelations:")
                for corr in outcome['correlations']:
                    print(f"- {corr}")
                    
            if outcome['notes']:
                print("\nDetailed Notes:")
                for note in outcome['notes']:
                    print(f"- {note}")
            
            # 5) Update top symbols tracker
            tracker.update(sym, patterns)
            
            # Print current rankings if this symbol made it to top 5%
            summary = tracker.get_summary()
            for category, top_symbols in summary['top_by_category'].items():
                if any(s['symbol'] == sym for s in top_symbols):
                    print(f"\n*** Symbol made it to top 5% in {category}! ***")
                    print("Current leaders in this category:")
                    for idx, s in enumerate(top_symbols[:5], 1):
                        print(f"{idx}. {s['symbol']}: {s['score']:.2f}")
            
            symbols_analyzed += 1
            
            # Adaptive sleep based on remaining API calls
            remaining = int(api.get_clock().next_open.timestamp() - time.time())
            if remaining < 300:  # Less than 5 minutes until reset
                sleep_time = 2.0  # Longer sleep when close to limit
            else:
                sleep_time = 0.5  # Normal operation
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Error analyzing {sym}: {str(e)}")
            time.sleep(1)  # Sleep longer on errors
            continue

    print("\nAnalysis Complete!")
    print(f"Successfully analyzed {symbols_analyzed} symbols")
    print(f"Total attempts: {attempts}")
    
    # Print comprehensive summary
    summary = tracker.get_summary()
    print(f"\nTotal symbols analyzed historically: {summary['total_analyzed']}")
    print("\nCurrent Leaders by Category:")
    
    for category, top_symbols in summary['top_by_category'].items():
        if top_symbols:
            print(f"\n{category.replace('_', ' ').title()}:")
            for idx, s in enumerate(top_symbols[:5], 1):
                print(f"{idx}. {s['symbol']}: {s['score']:.2f}")

if __name__ == "__main__":
    run_random_symbol_analysis(num_symbols=1000)
