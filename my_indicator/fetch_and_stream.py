#!/usr/bin/env python3

import time
import json
import requests
import pandas as pd
import threading
import websocket  # from websocket-client
from websocket import WebSocketApp

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
REST_API_KEY      = "7186AC48-4486-4B1B-95B9-74D1FB4788B1"
WEBSOCKET_API_KEY = "7186AC48-4486-4B1B-95B9-74D1FB4788B1"  # often the same as REST key
REST_BASE_URL     = "https://rest.coinapi.io/v1/ohlcv"
WS_BASE_URL       = "wss://ws.coinapi.io/v1/"

SYMBOLS = [
    "COINBASE_SPOT_BTC_USD",
    "COINBASE_SPOT_ETH_USD",
    "COINBASE_SPOT_LTC_USD",
    "COINBASE_SPOT_BCH_USD",
    "COINBASE_SPOT_LINK_USD",
    "COINBASE_SPOT_UNI_USD",
    "COINBASE_SPOT_AAVE_USD",
    "COINBASE_SPOT_SOL_USD",
    "COINBASE_SPOT_DOGE_USD",
    "COINBASE_SPOT_DOT_USD",
    "COINBASE_SPOT_AVAX_USD",
    "COINBASE_SPOT_SUSHI_USD"
]
# We'll assume we want 1,000 bars of 1-min OHLCV
# and then subscribe to 1-min updates via WebSocket
HISTORICAL_LIMIT  = 50
OHLCV_PERIOD      = "1MIN"

# ---------------------------------------------------------------------
# Fetch initial historical data (REST)
# ---------------------------------------------------------------------

def fetch_historical_ohlcv(symbol_id: str,
                           period_id: str = "1MIN",
                           limit: int = 1000) -> pd.DataFrame:
    """
    Fetch up to 'limit' bars of 1-min OHLCV data for 'symbol_id' from CoinAPI.
    Returns a DataFrame with columns [time_open, time_close, price_open, price_high, price_low, price_close, volume_traded].
    Adjust columns as needed. 
    """
    endpoint = f"{REST_BASE_URL}/{symbol_id}/history"
    headers  = {"X-CoinAPI-Key": REST_API_KEY}
    params   = {
        "period_id": period_id,
        "limit": limit
    }
    
    print(f"[REST] Fetching {limit} {period_id} bars for {symbol_id} ...")
    resp = requests.get(endpoint, headers=headers, params=params)
    if resp.status_code != 200:
        print(f"[ERROR] REST fetch for {symbol_id} failed: {resp.text}")
        return pd.DataFrame()  # empty
    
    data = resp.json()
    if not data:
        print(f"[WARN] No data returned for {symbol_id}")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    # Typical JSON fields from CoinAPI might be:
    # "time_period_start", "time_period_end", "time_open", "time_close",
    # "price_open", "price_high", "price_low", "price_close", "volume_traded", "trades_count"
    
    # We'll just keep columns we need:
    keep_cols = [
        "time_period_start",
        "time_period_end",
        "time_open",
        "time_close",
        "price_open",
        "price_high",
        "price_low",
        "price_close",
        "volume_traded"
    ]
    df = df[keep_cols]
    
    # Convert times to Pandas Timestamps
    df["time_period_start"] = pd.to_datetime(df["time_period_start"], utc=True)
    df["time_period_end"]   = pd.to_datetime(df["time_period_end"], utc=True)
    df["time_open"]         = pd.to_datetime(df["time_open"], utc=True)
    df["time_close"]        = pd.to_datetime(df["time_close"], utc=True)
    
    # Sort by start time ascending
    df.sort_values("time_period_start", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

# ---------------------------------------------------------------------
# WebSocket management
# ---------------------------------------------------------------------

class CoinAPIWebSocket:
    """
    A simple wrapper to:
    1) Connect to CoinAPI WebSocket
    2) Send 'hello' (or 'subscribe') message for OHLCV data
    3) Receive real-time bars & pass them to a callback
    """
    def __init__(self, api_key: str,
                 symbol_ids: list,
                 period_id: str,
                 on_ohlcv_callback):
        
        self.api_key       = api_key
        self.symbol_ids    = symbol_ids
        self.period_id     = period_id
        self.ws            = None
        self.running       = False
        self.on_ohlcv_cb   = on_ohlcv_callback  # function to handle new bars
    
    def _on_open(self, ws):
        print("[WS] Connected. Sending hello/subscribe message...")
        
        # "hello" overrides entire subscription scope. 
        # Alternatively, you can "subscribe" to add to existing subscription.
        # We'll do 'hello' to define exactly what we want:
        
        hello_msg = {
            "type": "hello",
            "apikey": self.api_key,
            "heartbeat": False,
            "subscribe_data_type": ["ohlcv"],
            # We only want 1-min bars:
            "subscribe_filter_period_id": [self.period_id],
            # We'll do exact matches for each symbol by appending "$"
            "subscribe_filter_symbol_id": [s + "$" for s in self.symbol_ids]
        }
        ws.send(json.dumps(hello_msg))
    
    def _on_message(self, ws, message):
        try:
            msg = json.loads(message)
        except Exception as e:
            print(f"[WS] Error decoding JSON: {e}")
            return
        
        # Two main message types we'll get for OHLCV:
        #  - "ohlcv": { "symbol_id": "...", "time_period_start": ..., "price_open": ..., etc. }
        #  - "error": { "type": "error", "message": "..." }
        msg_type = msg.get("type")
        
        if msg_type == "ohlcv":
            # This is a new bar. We'll pass it to our callback.
            # Example msg might have fields like:
            #  "symbol_id", "time_period_start", "time_period_end", "time_open", "time_close",
            #  "price_open", "price_high", "price_low", "price_close", "volume_traded", ...
            self.on_ohlcv_cb(msg)
        
        elif msg_type == "error":
            print(f"[WS] Error: {msg.get('message')}")
            # By default, CoinAPI closes the connection after sending an error.
            # So we can shut down or handle it as needed.
        
        elif msg_type == "heartbeat":
            # If heartbeat=True in hello, you'll get these every second or so
            pass
        
        else:
            # Possibly other message types
            # e.g. 'hello' response, 'unsubscribe' ack, etc.
            pass
    
    def _on_error(self, ws, error):
        print(f"[WS] WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_reason):
        print(f"[WS] Connection closed: {close_status_code} - {close_reason}")
    
    def start(self):
        print("[WS] Starting connection thread...")
        self.running = True
        self.ws = WebSocketApp(
            WS_BASE_URL,
            on_open    = self._on_open,
            on_message = self._on_message,
            on_error   = self._on_error,
            on_close   = self._on_close
        )
        
        # We’ll run this in a thread so it doesn’t block the main thread
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()
    
    def stop(self):
        print("[WS] Stopping WebSocket...")
        self.running = False
        if self.ws:
            self.ws.close()
        if self.thread.is_alive():
            self.thread.join()


# ---------------------------------------------------------------------
# Putting it all together
# ---------------------------------------------------------------------

class DataManager:
    """
    Manages historical DataFrames (one per symbol) and updates them
    with new bars from the WebSocket stream.
    """
    def __init__(self, symbols, period_id):
        self.symbols   = symbols
        self.period_id = period_id
        
        # We'll store each symbol's DataFrame in a dict
        # Key = symbol_id (like "COINBASE_SPOT_BTC_USD"), value = pd.DataFrame
        self.data = {}
        
        # We'll fetch initial data from REST
        for sym in self.symbols:
            df = fetch_historical_ohlcv(sym, period_id=self.period_id, limit=HISTORICAL_LIMIT)
            self.data[sym] = df
            print(f"[INIT] {sym} => got {len(df)} bars from REST")
        
        # Initialize WebSocket
        self.ws = CoinAPIWebSocket(
            api_key      = WEBSOCKET_API_KEY,
            symbol_ids   = self.symbols,
            period_id    = self.period_id,
            on_ohlcv_callback = self._on_ohlcv
        )
    
    def _on_ohlcv(self, msg: dict):
        """
        This callback is invoked each time a new OHLCV bar arrives from WebSocket.
        We'll parse it and append to the corresponding DataFrame.
        """
        symbol_id = msg.get("symbol_id")
        if not symbol_id:
            return
        
        # Extract relevant fields
        time_start = pd.to_datetime(msg.get("time_period_start"), utc=True)
        time_end   = pd.to_datetime(msg.get("time_period_end"),   utc=True)
        time_open  = pd.to_datetime(msg.get("time_open"),         utc=True)
        time_close = pd.to_datetime(msg.get("time_close"),        utc=True)
        
        price_open  = msg.get("price_open")
        price_high  = msg.get("price_high")
        price_low   = msg.get("price_low")
        price_close = msg.get("price_close")
        volume      = msg.get("volume_traded")
        
        # Build a one-row DataFrame
        new_row = pd.DataFrame([{
            "time_period_start": time_start,
            "time_period_end":   time_end,
            "time_open":         time_open,
            "time_close":        time_close,
            "price_open":        price_open,
            "price_high":        price_high,
            "price_low":         price_low,
            "price_close":       price_close,
            "volume_traded":     volume
        }])
        
        # Append to existing DF
        df = self.data.get(symbol_id)
        if df is None:
            # Possibly create a new DataFrame if symbol was not in self.symbols
            df = pd.DataFrame()
        
        # Concatenate & remove duplicates if needed
        df = pd.concat([df, new_row], ignore_index=True)
        df.drop_duplicates(subset=["time_period_start"], keep="last", inplace=True)
        df.sort_values("time_period_start", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Save back
        self.data[symbol_id] = df
        # For debugging:
        print(f"[WS] Got new bar for {symbol_id} => {time_start} | Close={price_close}")
    
    def start(self):
        """
        Start the WebSocket streaming process.
        """
        self.ws.start()
    
    def stop(self):
        """
        Stop the WebSocket streaming process.
        """
        self.ws.stop()

        # If you want, you can also save the final data to disk, etc.
        # for sym, df in self.data.items():
        #    df.to_csv(f"{sym}.csv", index=False)


def main():
    # Create the data manager which also fetches initial data
    manager = DataManager(SYMBOLS, OHLCV_PERIOD)
    
    # Start the WebSocket streaming
    manager.start()
    
    # Now we can just let it run; new bars will flow in.
    # We'll run for e.g. 10 minutes to demonstrate.
    try:
        while True:
            time.sleep(60)
            # at each minute, you might do more logic:
            #   e.g. run indicators, place trades, etc.
            #   manager.data[symbol_id] -> you have the updated DataFrame
    except KeyboardInterrupt:
        print("[MAIN] Shutting down...")
    finally:
        manager.stop()


if __name__ == "__main__":
    main()
