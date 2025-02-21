#!/usr/bin/env python3

import os
import time
import json
import threading
from datetime import datetime, timezone
import pandas as pd
import requests
import websocket  # from the "websocket-client" package
import alpaca_trade_api as tradeapi

from vwap_obv_strategy import VAMEStrategy
from config import config

###############################################################################
# CONFIG & CONSTANTS
###############################################################################

ALPACA_API_KEY     = os.getenv('ALPACA_API_KEY',  config.get('ALPACA_API_KEY'))
ALPACA_SECRET_KEY  = os.getenv('ALPACA_SECRET_KEY',config.get('ALPACA_SECRET_KEY'))
ALPACA_BASE_URL    = 'https://paper-api.alpaca.markets'

COINAPI_KEY       = "7186AC48-4486-4B1B-95B9-74D1FB4788B1" # Replace with your real key
COINAPI_WS_URL    = "wss://ws.coinapi.io/v1/"
COINAPI_REST_URL  = "https://rest.coinapi.io/v1/ohlcv"
PERIOD_ID         = "1MIN"
HISTORICAL_LIMIT  = 1000  # how many bars to fetch at startup
SYMBOLS           = [
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
LOOP_INTERVAL     = 60  # seconds

# If you also want to map "BTC/USD" -> "COINBASE_SPOT_BTC_USD", do that as well
# e.g. { "BTC/USD": "COINBASE_SPOT_BTC_USD", ... }
###########################################################################


###############################################################################
# DATA MANAGER: Fetch historical data + WebSocket streaming
###############################################################################
class DataManager:
    """
    Manages historical OHLCV data (1-min bars) for multiple symbols,
    and updates them in real-time from CoinAPI WebSocket.
    """
    def __init__(self, symbols, period_id, coinapi_key, historical_limit=1000):
        self.symbols          = symbols
        self.period_id        = period_id
        self.coinapi_key      = coinapi_key
        self.historical_limit = historical_limit
        
        # symbol -> DataFrame of bars
        self.data = {}
        
        # WebSocket management
        self.ws       = None
        self.ws_thread= None
        self.running  = False
        
        # 1) Fetch initial historical data
        for sym_id in self.symbols:
            df = self.fetch_historical(sym_id, self.period_id, self.historical_limit)
            self.data[sym_id] = df
            print(f"[INIT] {sym_id} => got {len(df)} bars from REST")

    def fetch_historical(self, symbol_id: str, period_id: str, limit: int) -> pd.DataFrame:
        """Fetch up to 'limit' 1-min bars for 'symbol_id' from CoinAPI via REST."""
        endpoint = f"{COINAPI_REST_URL}/{symbol_id}/history"
        headers  = {"X-CoinAPI-Key": self.coinapi_key}
        params   = {
            "period_id": period_id,
            "limit": limit
        }
        print(f"[REST] Fetching {limit} {period_id} bars for {symbol_id} ...")
        try:
            resp = requests.get(endpoint, headers=headers, params=params)
            if resp.status_code != 200:
                print(f"[ERROR] REST fetch failed ({resp.status_code}): {resp.text}")
                return pd.DataFrame()
        except Exception as e:
            print(f"[ERROR] fetch_historical() exception: {e}")
            return pd.DataFrame()
        
        data = resp.json()
        if not data:
            print(f"[WARN] No data returned for {symbol_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        if df.empty:
            return df
        
        # rename columns to standard
        df.rename(columns={
            "time_period_start": "time_period_start",
            "time_period_end":   "time_period_end",
            "time_open":         "time_open",
            "time_close":        "time_close",
            "price_open":        "price_open",
            "price_high":        "price_high",
            "price_low":         "price_low",
            "price_close":       "price_close",
            "volume_traded":     "volume_traded",
        }, inplace=True)
        
        # convert to datetime
        df["time_period_start"] = pd.to_datetime(df["time_period_start"], utc=True)
        df["time_period_end"]   = pd.to_datetime(df["time_period_end"],   utc=True)
        df["time_open"]         = pd.to_datetime(df["time_open"],         utc=True)
        df["time_close"]        = pd.to_datetime(df["time_close"],        utc=True)
        
        # sort ascending by start time
        df.sort_values("time_period_start", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    
    def start_stream(self):
        """Open the WebSocket and start receiving real-time bars."""
        print("[WS] Starting WebSocket connection...")
        self.running = True
        
        def on_open(ws):
            print("[WS] Connected. Sending hello message...")
            # We'll do "hello" with OHLCV subscription
            # We'll match each symbol exactly by appending "$" to filter
            symbol_filters = [s + "$" for s in self.symbols]
            hello_msg = {
                "type": "hello",
                "apikey": self.coinapi_key,
                "heartbeat": False,
                "subscribe_data_type": ["ohlcv"],
                "subscribe_filter_period_id": [self.period_id],
                "subscribe_filter_symbol_id": symbol_filters
            }
            ws.send(json.dumps(hello_msg))
        
        def on_message(ws, message):
            try:
                msg = json.loads(message)
            except Exception as e:
                print(f"[WS] JSON parse error: {e}")
                return
            
            if msg.get("type") == "ohlcv":
                self.on_ohlcv(msg)
            elif msg.get("type") == "error":
                print(f"[WS] Error: {msg.get('message')}")
            # else ignore heartbeats, etc.
        
        def on_close(ws, close_status, close_msg):
            print(f"[WS] Closed: {close_status} -> {close_msg}")
            self.running = False
        
        def on_error(ws, error):
            print(f"[WS] Error: {error}")
        
        self.ws = websocket.WebSocketApp(
            COINAPI_WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_close=on_close,
            on_error=on_error
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()

    def stop_stream(self):
        """Gracefully close the WebSocket."""
        self.running = False
        if self.ws:
            self.ws.close()
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join()
        print("[WS] Stopped.")

    def on_ohlcv(self, msg: dict):
        """Handle incoming real-time OHLCV bars, append/update DataFrame."""
        symbol_id = msg.get("symbol_id")
        if not symbol_id:
            return
        
        time_start = pd.to_datetime(msg["time_period_start"], utc=True)
        time_end   = pd.to_datetime(msg["time_period_end"],   utc=True)
        time_open  = pd.to_datetime(msg["time_open"],         utc=True)
        time_close = pd.to_datetime(msg["time_close"],        utc=True)
        
        price_open  = msg["price_open"]
        price_high  = msg["price_high"]
        price_low   = msg["price_low"]
        price_close = msg["price_close"]
        volume      = msg["volume_traded"]
        
        df = self.data.get(symbol_id, pd.DataFrame())
        
        # Build a 1-row DF
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
        
        # Concatenate + deduplicate
        df = pd.concat([df, new_row], ignore_index=True)
        # Remove duplicates by "time_period_start"
        df.drop_duplicates(subset=["time_period_start"], keep="last", inplace=True)
        df.sort_values("time_period_start", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        self.data[symbol_id] = df
        #print(f"[WS] New bar -> {symbol_id} | {time_start} | close={price_close}")


###############################################################################
# LIVE TRADER: uses DataManager + your strategy
###############################################################################

class LiveTrader:
    def __init__(self):
        # Alpaca
        self.api = tradeapi.REST(
            key_id     = ALPACA_API_KEY,
            secret_key = ALPACA_SECRET_KEY,
            base_url   = ALPACA_BASE_URL,
            api_version='v2'
        )
        
        # Make sure you can connect
        try:
            account = self.api.get_account()
            print(f"Connected to Alpaca account: {account.id}")
            print(f"Account status: {account.status}")
            print(f"Equity:       ${float(account.equity):.2f}")
            print(f"Buying power: ${float(account.buying_power):.2f}")
        except Exception as e:
            print(f"[ERROR] Alpaca connection: {e}")
            raise
        
        # Strategy
        self.strategy = VAMEStrategy(config)
        
        # Data Manager (handles historical + live WebSocket)
        self.data_manager = DataManager(
            symbols          = SYMBOLS,
            period_id        = PERIOD_ID,
            coinapi_key      = COINAPI_KEY,
            historical_limit = HISTORICAL_LIMIT
        )
        
        # Position tracking
        self.positions      = {}
        self.last_trade_time= {}
        self.loop_interval  = LOOP_INTERVAL

    def start(self):
        """Begin streaming data + run the main trading loop."""
        self.data_manager.start_stream()
        print("[INIT] DataManager streaming started.")
        
        try:
            while True:
                self.update_positions()
                # We'll run the strategy for each symbol in SYMBOLS
                # using the up-to-date DataFrame from DataManager.
                
                for symbol_id in SYMBOLS:
                    df = self.data_manager.data.get(symbol_id, pd.DataFrame())
                    if len(df) < 50:
                        print(f"[INFO] Not enough bars yet for {symbol_id}, skipping.")
                        continue
                    
                    # Convert the last X bars into the format your strategy expects
                    # Typically you'll rename "price_open" -> "open", etc., or adapt
                    # For brevity, let's do a quick rename so the strategy sees columns:
                    # open, high, low, close, volume, etc.
                    
                    bars = df.copy()
                    bars.rename(columns={
                        "price_open":  "open",
                        "price_high":  "high",
                        "price_low":   "low",
                        "price_close": "close",
                        "volume_traded":"volume"
                    }, inplace=True)
                    bars.set_index("time_period_start", inplace=True)  # strategy expects time index?
                    bars.sort_index(inplace=True)
                    
                    # Precompute indicators
                    bars_ind = self.strategy.precompute_indicators(bars)
                    # Classify regime
                    bars_ind['regime'] = self.strategy.regime_classifier.classify(bars_ind)
                    # Generate signals
                    bars_ind = self.strategy.signal_generator.generate_signals(bars_ind)
                    print("[DEBUG] Last row for", symbol_id)
                    print(bars_ind.tail(1)[
                        ['close','vwap','obv','mfi','price_signal','obv_signal','mfi_signal','regime']
                    ])
                    
                    # Now run strategy => yields trades for the last bar
                    trades = self.strategy.run(bars_ind)
                    if not trades:
                        continue
                    
                    # Place trades with Alpaca
                    self.execute_trades(symbol_id, trades)
                
                print(f"[SLEEP] Sleeping {self.loop_interval} seconds...")
                time.sleep(self.loop_interval)
        
        except KeyboardInterrupt:
            print("[STOP] KeyboardInterrupt, shutting down.")
        except Exception as e:
            print(f"[ERROR] Main loop: {e}")
        finally:
            self.data_manager.stop_stream()
            print("[SHUTDOWN] Exiting.")

    def update_positions(self):
        """Refresh local position info from Alpaca."""
        try:
            open_positions = self.api.list_positions()
            positions_dict = {}
            for pos in open_positions:
                positions_dict[pos.symbol] = {
                    'qty':          float(pos.qty),
                    'entry_price':  float(pos.avg_entry_price),
                    'current_price':float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl':float(pos.unrealized_pl)
                }
            self.positions = positions_dict
        except Exception as e:
            print(f"[ERROR] update_positions: {e}")

    def execute_trades(self, symbol_id, trades):
        """
        Given a list of trades from the strategy, place them on Alpaca if conditions are met.
        The 'symbol_id' is the CoinAPI symbol, but for Alpaca we might need something like "BTCUSD" if supported.
        """
        # Convert "COINBASE_SPOT_BTC_USD" -> "BTCUSD" if necessary
        # For example, just remove "COINBASE_SPOT_" and "_USD"
        # Or do a dictionary map if you have one
        alpaca_symbol = symbol_id.replace("COINBASE_SPOT_", "").replace("_USD", "USD")
        
        current_pos = self.positions.get(alpaca_symbol)  # might be None
        current_qty = current_pos['qty'] if current_pos else 0.0
        
        for t in trades:
            action   = t['action']
            size     = t['size']
            # you might have "pnl", "reason", "regime" in 't' as well
            
            # optional cooldown
            last_time = self.last_trade_time.get(alpaca_symbol, None)
            if last_time is not None:
                minutes_since = (datetime.now(timezone.utc) - last_time).total_seconds()/60.0
                if minutes_since < 5:
                    print(f"[INFO] Skip {symbol_id} trade, too soon. {minutes_since:.1f}m < 5m")
                    continue
            
            if action == "BUY":
                if current_qty < 0:
                    # Exit short
                    qty_to_buy = abs(current_qty)
                    if self.submit_order(alpaca_symbol, qty_to_buy, "buy"):
                        print(f"[TRADE] Covered short for {alpaca_symbol}")
                        self.last_trade_time[alpaca_symbol] = datetime.now(timezone.utc)
                elif current_qty == 0:
                    # New long
                    if self.submit_order(alpaca_symbol, size, "buy"):
                        print(f"[TRADE] Entered new long for {alpaca_symbol}")
                        self.last_trade_time[alpaca_symbol] = datetime.now(timezone.utc)
                else:
                    print("[INFO] Already in a long, ignoring BUY.")
            
            elif action == "SELL":
                if current_qty > 0:
                    # Exit long
                    if self.submit_order(alpaca_symbol, current_qty, "sell"):
                        print(f"[TRADE] Exited long for {alpaca_symbol}")
                        self.last_trade_time[alpaca_symbol] = datetime.now(timezone.utc)
                elif current_qty == 0:
                    # New short
                    if self.submit_order(alpaca_symbol, size, "sell"):
                        print(f"[TRADE] Opened new short for {alpaca_symbol}")
                        self.last_trade_time[alpaca_symbol] = datetime.now(timezone.utc)
                else:
                    print("[INFO] Already in a short, ignoring SELL.")

    def submit_order(self, alpaca_symbol, qty, side) -> bool:
        """Places a market order on Alpaca."""
        try:
            self.api.submit_order(
                symbol        = alpaca_symbol,
                qty           = qty,
                side          = side,
                type          = 'market',
                time_in_force = 'gtc'
            )
            print(f"[ORDER] Submitted {side} order for {qty} {alpaca_symbol}")
            return True
        except Exception as e:
            print(f"[ERROR] submit_order for {alpaca_symbol}: {e}")
            return False


###############################################################################
# MAIN ENTRY
###############################################################################
if __name__ == "__main__":
    trader = LiveTrader()
    trader.start()
