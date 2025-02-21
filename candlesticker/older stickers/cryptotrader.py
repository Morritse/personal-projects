import os
import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta, timezone

# We'll import your existing analysis class or code logic:
# from cryptosticker import IntradayCryptoAnalyzer
from cryptosticker import IntradayCryptoAnalyzer  # adjust import as needed

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_PAPER_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_PAPER_SECRET")
BASE_URL = "https://paper-api.alpaca.markets"

CRYPTO_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD"]
TOTAL_CAPITAL = 50000.0  # We'll fix total capital at $50k

# How often to re-check signals (in seconds); 300s = 5 minutes
REBALANCE_INTERVAL = 300  

class LiveRebalancer:
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=BASE_URL,
            api_version='v2'
        )
        self.analyzer = IntradayCryptoAnalyzer()  # from your code
        # We'll remove the slash for Alpaca's symbol format
        self.symbol_map = {
            "BTC/USD": "BTCUSD",
            "ETH/USD": "ETHUSD",
            "SOL/USD": "SOLUSD",
            "DOGE/USD": "DOGEUSD"
        }

    def run_live_cycle(self):
        """
        1) For each symbol, run analysis => final_signal
        2) Compute weight => target notional
        3) Rebalance positions to match target notional
        """
        signals = {}
        
        # 1) Gather final_signals for each symbol
        for sym in CRYPTO_SYMBOLS:
            final_signal = self.compute_signal_for_symbol(sym)
            signals[sym] = final_signal

        # 2) Convert negative signals to 0, since we skip shorting
        positive_signals = {sym: max(0, s) for sym, s in signals.items()}
        sum_signals = sum(positive_signals.values())

        # If sum_signals == 0, no positions
        if sum_signals == 0:
            # Liquidate all positions in these symbols
            print("All final signals <= 0 => Liquidating positions.")
            self.liquidate_all_positions()
            return

        # Otherwise, compute each symbol's fraction
        for sym in CRYPTO_SYMBOLS:
            symbol_api = self.symbol_map[sym]
            sig = positive_signals[sym]
            weight = sig / sum_signals if sum_signals != 0 else 0
            target_notional = weight * TOTAL_CAPITAL
            self.rebalance_symbol(sym, symbol_api, target_notional)

    def compute_signal_for_symbol(self, sym: str) -> float:
        """
        Runs your IntradayCryptoAnalyzer for that symbol
        and returns the *most recent* final_signal.
        """
        # We'll run analysis. This fetches historical data,
        # resamples, etc. Then we pick the last bar's final_signal
        self.analyzer.get_latest_signal(sym)  
        
        # The run_analysis prints bars but let's assume we store the result:
        df_res = self.analyzer.df_res  # Suppose we store the final DataFrame inside the analyzer
        if df_res.empty:
            return 0
        
        # final_signal is in df_res['final_signal']
        # We'll take the last known bar
        last_row = df_res.iloc[-1]
        final_sig = last_row['final_signal']
        return final_sig

    def rebalance_symbol(self, sym: str, symbol_api: str, target_notional: float):
        """
        Adjust the position for symbol_api to match 'target_notional' in USD.
        """
        # 1) Get current price
        last_quote = self.get_current_price(symbol_api)
        if last_quote is None:
            print(f"Cannot get price for {symbol_api}, skipping rebalance.")
            return
        
        # 2) Desired units = target_notional / price
        desired_qty = 0
        if last_quote > 0:
            desired_qty = target_notional / last_quote
        
        # 3) Check current position
        current_position = self.get_position_qty(symbol_api)
        delta = desired_qty - current_position
        
        # If delta is > 0 => we need to buy more
        # If delta < 0 => we need to sell
        if abs(delta) < 1e-6:
            print(f"{sym} already in correct position.")
            return
        
        if delta > 0:
            # We buy delta units at market
            notional_delta = delta * last_quote
            print(f"[REB] Buying +${notional_delta:.2f} of {sym} => delta qty {delta:.6f}")
            try:
                self.api.submit_order(
                    symbol=symbol_api,
                    notional=notional_delta,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            except Exception as e:
                print(f"Buy error: {e}")

        else:
            # We sell abs(delta) units
            notional_delta = abs(delta) * last_quote
            print(f"[REB] Selling -${notional_delta:.2f} of {sym} => delta qty {delta:.6f}")
            try:
                self.api.submit_order(
                    symbol=symbol_api,
                    notional=notional_delta,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            except Exception as e:
                print(f"Sell error: {e}")

    def get_current_price(self, symbol_api: str) -> float:
        """
        A quick approach: use the last trade or last quote from Alpaca's crypto feed.
        Or you can parse your 1-min bar data. We'll do a naive approach:
        """
        try:
            # get_latest_trade returns a trade object with .price
            t = self.api.get_latest_trade(symbol_api)
            return float(t.price)
        except Exception as e:
            print(f"Error getting current price for {symbol_api}: {e}")
            return None

    def get_position_qty(self, symbol_api: str) -> float:
        """
        Return how many units of symbol_api we currently hold in the paper account.
        If none, or no position, return 0.
        """
        try:
            position = self.api.get_position(symbol_api)
            qty = float(position.qty)
            return qty
        except tradeapi.rest.APIError as e:
            # if "position does not exist" => 0
            if "does not exist" in str(e).lower():
                return 0.0
            else:
                print(f"Error fetching position for {symbol_api}: {e}")
                return 0.0

    def liquidate_all_positions(self):
        """
        Liquidate all CRYPTO_SYMBOLS positions (only if we hold them).
        """
        for sym in CRYPTO_SYMBOLS:
            symbol_api = self.symbol_map[sym]
            qty = self.get_position_qty(symbol_api)
            if qty > 0:
                last_price = self.get_current_price(symbol_api)
                notional = qty * last_price
                print(f"[LIQUIDATE] Selling all {sym}: ${notional:.2f} notional.")
                try:
                    self.api.submit_order(
                        symbol=symbol_api,
                        notional=notional,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                except Exception as e:
                    print(f"Liquidation error for {sym}: {e}")

def main_loop():
    rebalancer = LiveRebalancer()

    while True:
        print(f"\n[{datetime.now()}] Starting 5-minute rebalance cycle...")
        rebalancer.run_live_cycle()
        print(f"[{datetime.now()}] Rebalance done. Sleeping {REBALANCE_INTERVAL} seconds...\n")
        time.sleep(REBALANCE_INTERVAL)

if __name__ == "__main__":
    main_loop()
