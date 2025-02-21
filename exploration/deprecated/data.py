import os
import requests
import datetime
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

##############################################################################
# 1) ENVIRONMENT CONFIG
##############################################################################

load_dotenv()  # Load variables from .env if present

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# For PAPER environment, not real money
ALPACA_DATA_URL = "https://data.alpaca.markets"  # Alpaca Market Data endpoint

# Symbols you want to loop over individually
SYMBOLS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA",
    "SPY","QQQ","IWM",
    "AMD","COIN","PLTR","NET","CRWD","DDOG",
    "NIO","RIVN","LCID","SOFI","RBLX"
]

# Alpaca v2 Market Data parameters
TIMEFRAME   = "1Min"   # e.g. 1Min, 5Min, 15Min, 1Hour, 1Day, etc.
FEED        = "sip"    # "sip", "iex", or "otc"
ADJUSTMENT  = "raw"    # "raw", "split", "dividends", or "all"

##############################################################################
# 2) DATE RANGE: "TODAY" LOOKBACK
##############################################################################
# We'll look back 3 days from now (in UTC). Adjust as needed.

LOOKBACK_DAYS = 3

# Get current time in PST
pst = ZoneInfo('America/Los_Angeles')
now_pst = datetime.datetime.now(pst)

# Calculate start/end in PST, including extended hours
# Extended hours: 4:00 AM - 5:00 PM PST (7:00 AM - 8:00 PM EST)
end_dt = now_pst.replace(microsecond=0, second=0)
if end_dt.hour < 17:  # Before extended hours end
    end_dt = end_dt.replace(hour=17, minute=0)  # Set to extended hours end
else:
    end_dt = end_dt.replace(hour=17, minute=0)  # Set to extended hours end

# Go back LOOKBACK_DAYS trading days
start_dt = end_dt - timedelta(days=LOOKBACK_DAYS)
start_dt = start_dt.replace(hour=4, minute=0)  # Set to extended hours start

# Market hours for reference (PST):
# Pre-market: 4:00 AM - 6:30 AM
# Regular: 6:30 AM - 1:00 PM
# After-hours: 1:00 PM - 5:00 PM

# Convert to UTC for API
end_str = end_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
start_str = start_dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

print(f"Fetching data from {start_dt} PST to {end_dt} PST")

# For example:
# start_str = "2024-01-05T00:00:00Z"
# end_str   = "2024-01-08T14:15:35Z"
##############################################################################


def fetch_historical_bars_for_symbol(
    symbol: str,
    timeframe: str = "1Min",
    start: str = None,
    end: str = None,
    feed: str = "sip",
    adjustment: str = "raw",
    limit: int = 10000  # Up to 10,000 per docs
):
    """
    Fetch historical bars for a single symbol via Alpaca Market Data v2.
    Returns the parsed JSON (dict).
    """
    url = f"{ALPACA_DATA_URL}/v2/stocks/bars"
    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Accept": "application/json"
    }
    params = {
        "symbols": symbol,     # single symbol
        "timeframe": timeframe,
        "feed": feed,
        "adjustment": adjustment,
        "start": start,
        "end": end,
        "limit": limit,
    }

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    return r.json()  # typically includes {"bars": {"AAPL": [...], ...}, "next_page_token": ...}


def main():
    """
    Loops through all symbols individually, fetching data for each.
    Prints a quick summary for demonstration.
    """
    # For demonstration, store everything in a dict:
    # data_store = { symbol: bars_list, ... }
    data_store = {}

    for symbol in SYMBOLS:
        print(f"\n[INFO] Fetching data for {symbol}...")
        json_resp = fetch_historical_bars_for_symbol(
            symbol,
            timeframe=TIMEFRAME,
            start=start_str,
            end=end_str,
            feed=FEED,
            adjustment=ADJUSTMENT,
            limit=10000  # large enough for multiple days of 1-min bars
        )

        # Expecting the structure:
        # {
        #   "bars": {
        #     "AAPL": [
        #       { "t": "...", "o":123.45, "h":..., "l":..., "c":..., "v":... },
        #       ...
        #     ]
        #   },
        #   "next_page_token": ...
        # }
        bars_dict = json_resp.get("bars", {})
        bars_list = bars_dict.get(symbol, [])

        data_store[symbol] = bars_list

        # Basic demonstration: print how many bars we got
        # Convert UTC strings to PST for display
        start_utc = datetime.datetime.strptime(start_str, '%Y-%m-%dT%H:%M:%SZ')
        end_utc = datetime.datetime.strptime(end_str, '%Y-%m-%dT%H:%M:%SZ')
        start_utc = start_utc.replace(tzinfo=timezone.utc)
        end_utc = end_utc.replace(tzinfo=timezone.utc)
        start_pst = start_utc.astimezone(pst)
        end_pst = end_utc.astimezone(pst)
        
        print(f"  Received {len(bars_list)} bars")
        print(f"  Time range (PST): {start_pst.strftime('%Y-%m-%d %I:%M %p')} to {end_pst.strftime('%Y-%m-%d %I:%M %p')}")

        # If there is more data than 'limit', you would handle "pagination" 
        # by checking json_resp['next_page_token'] and calling the same 
        # endpoint with page_token=... in a loop. For now, ignoring it.

    # data_store now contains all the bars for each symbol over the last 3 days
    # Example: 
    #   data_store["AAPL"] = [ { "t":..., "o":..., ... }, ... ]

    # Just to show an example snippet for the last bar:
    for sym in data_store:
        bars = data_store[sym]
        if bars:
            # Convert UTC timestamp to PST and add session info
            last_bar = bars[-1].copy()  # Make a copy to avoid modifying original data
            bar_time_utc = datetime.datetime.strptime(last_bar['t'], '%Y-%m-%dT%H:%M:%SZ')
            bar_time_utc = bar_time_utc.replace(tzinfo=timezone.utc)
            bar_time_pst = bar_time_utc.astimezone(pst)
            
            # Add session info
            hour = bar_time_pst.hour
            minute = bar_time_pst.minute
            time_str = bar_time_pst.strftime('%Y-%m-%d %H:%M:%S PST')
            if hour < 4 or (hour == 4 and minute == 0):
                time_str += " (After Hours)"
            elif hour < 6 or (hour == 6 and minute < 30):
                time_str += " (Pre-Market)"
            elif hour < 13 or (hour == 13 and minute == 0):
                time_str += " (Market Hours)"
            else:
                time_str += " (After Hours)"
                
            last_bar['t'] = time_str
            print(f"\nSample for {sym}: Last bar => {last_bar}")
        else:
            print(f"\nNo data found for {sym} in this date range.")


if __name__ == "__main__":
    main()
