import os
import requests
import datetime
import json
from datetime import timedelta, timezone
import pandas as pd
import numpy as np

##############################################################################
# ENV & DATA FETCHING
##############################################################################

# ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
# ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_API_KEY="PKB47VPLE9OYP875O68E"
ALPACA_SECRET_KEY="cVNZxQ7LbGlYu5zhqCMTkTVgKKlatOjM4BY3Y3Pv"
ALPACA_DATA_URL = "https://data.alpaca.markets"  # Alpaca Market Data endpoint

SYMBOLS = [
    # Mega-Cap Tech / Growth
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","AMD","CRM","ORCL",
    # Additional Tech & High-Fliers
    "PYPL","ADBE","INTC","NFLX",
    # Major ETFs
    "SPY","QQQ","DIA","IWM",
    # Sector ETFs
    "XLF","XLK","XLE","XLV","XLY","XLI","XLB",
    # Financials
    "JPM","BAC","WFC",
    # Energy
    "XOM","CVX",
    # Healthcare
    "JNJ","PFE","UNH",
    # Consumer Defensive / Staples
    "WMT","PG","KO",
    # Consumer Discretionary
    "MCD","SBUX","HD",
    # Popular High-Volume / Momentum Names
    "COIN","PLTR","RIVN","SOFI","RBLX","LCID"
]

TIMEFRAME = "5Min"
FEED = "sip"
ADJUSTMENT = "raw"

LOOKBACK_DAYS = 3  # how many days from now (UTC) to look back
MAX_BARS_PER_REQ = 10000

def fetch_historical_bars_for_symbol(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    feed: str = "sip",
    adjustment: str = "raw",
    limit: int = 10000
):
    url = f"{ALPACA_DATA_URL}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "feed": feed,
        "adjustment": adjustment,
        "start": start,
        "end": end,
        "limit": limit
    }
    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()  # will throw HTTPError for 4xx/5xx
    return r.json()

##############################################################################
# INDICATOR CALCULATIONS
##############################################################################

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_donchian(df: pd.DataFrame, window: int = 20):
    df['donchian_upper'] = df['high'].rolling(window).max()
    df['donchian_lower'] = df['low'].rolling(window).min()
    return df

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    df['prev_close'] = df['close'].shift(1)
    df['tr0'] = df['high'] - df['low']
    df['tr1'] = (df['high'] - df['prev_close']).abs()
    df['tr2'] = (df['low'] - df['prev_close']).abs()
    tr = df[['tr0','tr1','tr2']].max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

##############################################################################
# SIGNAL LOGIC
##############################################################################

def get_continuous_signal(df: pd.DataFrame) -> float:
    """
    Return a floating signal in [-1..+1], 
    factoring in 'exit negativity' or bullish momentum.
    This is just an example approach.
    """
    # We'll examine the *most recent row*
    last = df.iloc[-1]

    close = last['close']
    ema9  = last['ema9']
    ema20 = last['ema20']
    rsi   = last['rsi']
    upper = last['donchian_upper']
    lower = last['donchian_lower']

    # 1) Check exit negativity
    # If close < ema9 => negative. How negative?
    # Let's say difference% = (ema9 - close)/close => scale in [-1..0]
    if close < ema9:
        difference = (ema9 - close) / close
        # clamp to 1 for large differences
        if difference > 1:
            difference = 1
        # Make signal negative
        return -difference

    # 2) Otherwise, compute a bullish strength in [0..+1]
    #    e.g. combine RSI, relative close above ema20, and 
    #    how high we are in the Donchian channel
    scaled_rsi = (rsi - 50) / 50  # RSI=50 =>0; RSI=100 =>+1; RSI=0 =>-1
    diff_ema20 = (close - ema20) / ema20 if ema20 > 0 else 0

    # For Donchian, fraction between lower..upper
    if pd.notna(upper) and pd.notna(lower) and (upper - lower) != 0:
        frac_in_channel = (close - lower) / (upper - lower)  # [0..1]
        scaled_channel  = 2 * frac_in_channel - 1            # [0..1] => [-1..+1]
    else:
        scaled_channel = 0

    # Weighted sum => clamp to [-1..+1]
    raw_signal = (0.5 * scaled_rsi) + (0.3 * diff_ema20) + (0.2 * scaled_channel)

    # Force range
    if raw_signal > 1:
        raw_signal = 1
    if raw_signal < -1:
        raw_signal = -1

    return raw_signal

##############################################################################
# MAIN
##############################################################################

def main():
    now_utc = datetime.datetime.now(timezone.utc)
    start_dt = now_utc - timedelta(days=LOOKBACK_DAYS)

    # Ensure we don't go beyond "now" (no future requests)
    # Just in case system clock is off or user changed date
    if start_dt > now_utc:
        start_dt = now_utc

    # Format UTC timestamps for API (removing timezone info and adding Z)
    start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Convert UTC to PST for display
    pst = datetime.timezone(datetime.timedelta(hours=-8))  # PST is UTC-8
    start_pst = start_dt.astimezone(pst)
    now_pst = now_utc.astimezone(pst)
    
    print(f"Fetching data from {start_pst.strftime('%Y-%m-%d %I:%M %p')} PST to {now_pst.strftime('%Y-%m-%d %I:%M %p')} PST\n")

    final_signals = {}

    for symbol in SYMBOLS:
        print(f"Fetching {TIMEFRAME} bars for {symbol}...")

        try:
            json_resp = fetch_historical_bars_for_symbol(
                symbol=symbol,
                timeframe=TIMEFRAME,
                start=start_str,
                end=end_str,
                feed=FEED,
                adjustment=ADJUSTMENT,
                limit=MAX_BARS_PER_REQ
            )
        except requests.HTTPError as e:
            # If we get 400 or 404 from the server, print message, skip
            print(f"  [Error] {e}")
            final_signals[symbol] = 0.0
            continue

        bars_dict = json_resp.get("bars", {})
        bars_list = bars_dict.get(symbol, [])

        if not bars_list:
            print(f"  No bars returned for {symbol}. Setting signal=0.")
            final_signals[symbol] = 0.0
            continue

        df = pd.DataFrame(bars_list)
        df.rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"},
                  inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df.sort_values("timestamp", inplace=True)

        # Indicators
        df['ema9']  = compute_ema(df['close'], 9)
        df['ema20'] = compute_ema(df['close'], 20)
        df['rsi']   = compute_rsi(df['close'], 7)
        df = compute_donchian(df, 20)
        df['atr']   = compute_atr(df, 14)

        # If there's not enough data for an indicator window, partial rows may be NaN
        # We'll drop them so the last row is guaranteed valid. 
        df.dropna(inplace=True)

        if df.empty:
            print(f"  After dropping NaNs, no data left for {symbol}. => 0")
            final_signals[symbol] = 0.0
            continue

        # Compute continuous signal
        continuous_signal = get_continuous_signal(df)
        final_signals[symbol] = float(np.round(continuous_signal, 4))  # round to 4 decimals

        print(f"  => final signal: {continuous_signal:.4f}")

    # Write to JSON
    with open("signals.json", "w") as f:
        json.dump(final_signals, f, indent=4)

    print("\nAll signals written to signals.json:")
    print(final_signals)

if __name__ == "__main__":
    main()
