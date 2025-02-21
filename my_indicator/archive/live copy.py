import os
import time
from datetime import datetime, timedelta
import pandas as pd
import alpaca_trade_api as tradeapi

# Assuming you have these local modules in the same folder:
from vwap_obv_strategy import VAMEStrategy
from config import config  # Contains your strategy parameters, including 'SYMBOLS'


# Retrieve Alpaca credentials from environment or config
ALPACA_API_KEY    = os.getenv('ALPACA_API_KEY',    config.get('ALPACA_API_KEY'))
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', config.get('ALPACA_SECRET_KEY'))
ALPACA_BASE_URL   = 'https://paper-api.alpaca.markets'  # Paper trading

class LiveTrader:
    def __init__(self):
        # 1) Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id     = ALPACA_API_KEY,
            secret_key = ALPACA_SECRET_KEY,
            base_url   = ALPACA_BASE_URL,
            api_version='v2'
        )
        
        # 2) Load your strategy with the config
        self.strategy = VAMEStrategy(config)
        
        # 3) Retrieve symbols from config (or fallback)
        self.symbols         = [
            'BTC/USD',
            'ETH/USD',
            'LTC/USD',
            'BCH/USD',
            'LINK/USD',
            'UNI/USD',
            'AAVE/USD',
            'SOL/USD',
            'DOGE/USD',
            'DOT/USD',
            'AVAX/USD',
            'SUSHI/USD'
        ]
        self.position_size   = 100_000  # Capital for position sizing or reference
        self.positions       = {}       # Track current positions as reported by Alpaca
        
        # 4) Verify account
        try:
            account = self.api.get_account()
            print(f"Connected to Alpaca account: {account.id}")
            print(f"Account status: {account.status}")
            print(f"Equity:       ${float(account.equity):.2f}")
            print(f"Buying power: ${float(account.buying_power):.2f}")
        except Exception as e:
            print(f"Error connecting to Alpaca: {e}")
            raise

    def _rfc3339_timestamp(self, dt: datetime) -> str:
        """
        Convert Python datetime to RFC3339 for Alpaca crypto bars.
        Example: '2025-01-07T23:33:47Z'
        """
        return dt.strftime('%Y-%m-%dT%H:%M:%SZ')

    def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch 1-minute historical data from Alpaca Crypto API,
        enough for our largest indicator lookbacks from config.
        """
        # 1) Determine largest lookback from your config parameters
        lookback = max(
            config['Historical Window'][0],  # e.g. 2500
            config['Current Window'][0],     # e.g. 500
            config['VWAP Window'][0],        # e.g. 350
            config['MFI Period'][0],         # e.g. 150
            config['trend_params']['ema_long_span'][0]  # e.g. 135
        )
        # Add extra buffer
        lookback = int(lookback * 1.25)  # 25% buffer

        # 2) Date range
        end_dt   = datetime.now()
        start_dt = end_dt - timedelta(minutes=lookback)
        
        bars_all = []
        current_end = end_dt

        # We'll fetch in ~7-day chunks to avoid time or bar-limit issues
        while current_end > start_dt:
            chunk_start = max(start_dt, current_end - timedelta(days=7))
            start_str   = self._rfc3339_timestamp(chunk_start)
            end_str     = self._rfc3339_timestamp(current_end)
            
            print(f"[INFO] Fetching {symbol} data from {chunk_start} to {current_end}")

            try:
                chunk = self.api.get_crypto_bars(
                    symbol,
                    timeframe = tradeapi.TimeFrame.Minute,
                    start     = start_str,
                    end       = end_str
                ).df
            except Exception as e:
                print(f"[WARN] Error fetching chunk: {e}")
                break

            if chunk.empty:
                print(f"[INFO] No data returned in chunk {chunk_start} -> {current_end}")
                break

            bars_all.append(chunk)
            current_end = chunk_start - timedelta(minutes=1)

        if not bars_all:
            print(f"[WARN] No data received at all for {symbol}")
            return pd.DataFrame()

        # 3) Combine the chunk data
        bars = pd.concat(bars_all[::-1])  # Reverse the list so earliest is first
        bars = bars.sort_index()
        bars = bars[~bars.index.duplicated(keep='first')]

        print(f"[INFO] Got {len(bars)} bars for {symbol}.")
        
        # 4) Format into OHLCV DataFrame
        df = pd.DataFrame({
            'open':   bars['open'],
            'high':   bars['high'],
            'low':    bars['low'],
            'close':  bars['close'],
            'volume': bars['volume']
        }, index=bars.index)
        
        # Ensure UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        return df

    def update_positions(self) -> None:
        """
        Refresh local knowledge of positions from Alpaca
        so we know how many shares of each symbol we hold, etc.
        """
        try:
            raw_positions = self.api.list_positions()
            positions_dict = {}
            for p in raw_positions:
                # p.symbol might be e.g. "ETHUSD"
                positions_dict[p.symbol] = {
                    'qty':           float(p.qty),
                    'entry_price':   float(p.avg_entry_price),
                    'current_price': float(p.current_price),
                    'market_value':  float(p.market_value),
                    'unreal_pl':     float(p.unrealized_pl)
                }
            self.positions = positions_dict
        except Exception as e:
            print(f"[ERROR] Could not update positions: {e}")

    def submit_order(self, symbol: str, qty: float, side: str) -> bool:
        """
        Submit a MARKET order. For crypto, you can do fractional qty if needed.
        """
        try:
            alpaca_symbol = symbol.replace('/', '')  # e.g. "BTCUSD"
            
            # You might want to check if 'qty' is within min order size 
            # or if you'd prefer notional=... 
            self.api.submit_order(
                symbol        = alpaca_symbol,
                qty           = qty,
                side          = side,  # 'buy' or 'sell'
                type          = 'market',
                time_in_force = 'gtc'
            )
            
            print(f"[ORDER] Submitted {side.upper()} for {qty} {symbol} at MARKET.")
            return True
        except Exception as e:
            print(f"[ERROR] submit_order failed for {symbol}: {e}")
            return False

    def log_trade(self, trade: dict, symbol: str) -> None:
        """
        Log the trade action (entry/exit) to a timestamped file for audit.
        """
        now_str   = datetime.now().strftime('%Y%m%d')  # daily file
        filename  = f"trades_{now_str}.log"
        with open(filename, 'a') as f:
            f.write("--------------------------------------------------\n")
            f.write(f"Time: {trade['timestamp']}\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Action: {trade['action']}\n")
            f.write(f"Price: {trade['price']:.2f}\n")
            f.write(f"Size: {trade['size']}\n")
            if 'pnl' in trade:
                f.write(f"PnL: {trade['pnl']:.2f}\n")
            if 'reason' in trade:
                f.write(f"Reason: {trade['reason']}\n")
            if 'regime' in trade:
                f.write(f"Regime: {trade['regime']}\n")
            if 'stop_loss' in trade:
                f.write(f"Stop Loss: {trade['stop_loss']:.2f}\n")
            if 'take_profit' in trade:
                f.write(f"Take Profit: {trade['take_profit']:.2f}\n")
            f.write("\n")

    def run(self):
        """
        Main infinite loop:
         1) Update our positions from Alpaca
         2) For each symbol, fetch data, run the strategy, possibly place an order
         3) Sleep ~60 seconds
        """
        print("[INIT] Starting live trading with 1-min data.")
        print(f"[INIT] Managing symbols: {self.symbols}")
        
        while True:
            try:
                # 1) Update local positions
                self.update_positions()
                
                # 2) Process each symbol
                for symbol in self.symbols:
                    print(f"\n[LOOP] Processing {symbol} ...")
                    df = self.get_historical_data(symbol)
                    if df.empty:
                        print(f"[SKIP] No data for {symbol}")
                        continue
                    
                    # 2a) Run strategy on the fresh data
                    trades = self.strategy.run(df)
                    if not trades:
                        print("[INFO] No new trade signals from strategy.")
                        continue
                    
                    # 2b) The last trade is presumably the newest event
                    latest_trade = trades[-1]
                    # Convert e.g. "ETH/USD" => "ETHUSD" for position dict
                    alpaca_symbol = symbol.replace('/', '')
                    
                    # 2c) Check if we already hold a position
                    current_pos = self.positions.get(alpaca_symbol, {})
                    current_qty = current_pos.get('qty', 0.0)
                    
                    # 2d) Compare the strategy's recommended trade with our current position
                    if latest_trade['action'] == 'BUY':
                        if current_qty <= 0:
                            # Means we have no position or a short => Buy
                            submitted = self.submit_order(symbol, latest_trade['size'], 'buy')
                            if submitted:
                                self.log_trade(latest_trade, symbol)
                        else:
                            print("[INFO] Already in a long position, skipping BUY.")
                    
                    elif latest_trade['action'] == 'SELL':
                        if current_qty >= 0:
                            # Means no position or a long => Sell
                            submitted = self.submit_order(symbol, latest_trade['size'], 'sell')
                            if submitted:
                                self.log_trade(latest_trade, symbol)
                        else:
                            print("[INFO] Already in a short position, skipping SELL.")
                
                # 3) Wait 60s
                print("[SLEEP] Sleeping 60s until next cycle.")
                time.sleep(60)
            
            except KeyboardInterrupt:
                print("\n[STOP] Keyboard interrupt, shutting down.")
                break
            except Exception as e:
                print(f"[ERROR] Main loop encountered exception: {e}")
                time.sleep(60)

if __name__ == '__main__':
    trader = LiveTrader()
    trader.run()
