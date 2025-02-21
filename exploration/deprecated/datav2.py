import os
import requests
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_DATA_URL = "https://data.alpaca.markets"  # Alpaca Market Data endpoint

def fetch_historical_bars_for_symbol(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    feed: str = "sip",
    adjustment: str = "raw",
    limit: int = 10000
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
        "symbols": symbol,
        "timeframe": timeframe,
        "feed": feed,
        "adjustment": adjustment,
        "start": start,
        "end": end,
        "limit": limit,
    }

    r = requests.get(url, headers=headers, params=params)
    r.raise_for_status()
    return r.json()  # typically includes {"bars": { symbol: [ {t,o,h,l,c,v}, ...] }, "next_page_token": ...}
