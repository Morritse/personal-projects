# cryptotrader.py

import os
import time
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
import requests
from datetime import datetime, timedelta, timezone

from stickerv2 import IntradayCryptoAnalyzer

# ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_PAPER_KEY")
# ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_PAPER_SECRET")
ALPACA_API_KEY="PK9GGP2I0PAS11A7IOQB"
ALPACA_SECRET_KEY="BcAYtWDkr11tpzCtBqC3Cud2VQ6VuXH8o3BH6Kg0"
BASE_URL = "https://paper-api.alpaca.markets"

CRYPTO_SYMBOLS = [
    "SHIB/USD", "ETH/USD", "BAT/USD", "BCH/USD", "BTC/USD", 
    "CRV/USD", "LTC/USD", "DOGE/USD", "YFI/USD", "DOT/USD",
    "GRT/USD", "LINK/USD", "MKR/USD", "SUSHI/USD", "UNI/USD",
    "USDC/USD", "AVAX/USD", "XTZ/USD", "USDT/USD", "AAVE/USD",
    "SOL/USD"
]
TOTAL_CAPITAL = 93934.38  
REBALANCE_INTERVAL = 30  # seconds (5 min)

class LiveRebalancer:
    def __init__(self):
        self.api = tradeapi.REST(
            key_id=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            base_url=BASE_URL,
            api_version='v2'
        )
        self.analyzer = IntradayCryptoAnalyzer()
        self.position_tracker = {}  # Track entry prices and costs
        self.total_fees = 0.0

    def run_live_cycle(self):
        signals = {}
        # 1) Get final_signal for each slash-based symbol
        for sym in CRYPTO_SYMBOLS:
            final_signal = self.analyzer.get_latest_signal(sym)
            signals[sym] = final_signal

        # 2) Flip signals and convert negative to 0 => skip short
        pos_signals = {s: max(0, val * -1) for s, val in signals.items()}
        sum_sigs = sum(pos_signals.values())

        if sum_sigs == 0:
            print("All signals <= 0, liquidate everything.")
            self.liquidate_all()
            return

        # Normalize weights across all positive signals
        pos_signals = {s: val/sum_sigs for s, val in pos_signals.items() if val > 0}

        # Get account balance and positions
        account = self.api.get_account()
        available_balance = float(account.cash)
        print(f"Available cash: ${available_balance:.2f}")
        
        # Calculate total portfolio value
        portfolio_value = available_balance
        print("Current positions:")
        for sym in CRYPTO_SYMBOLS:
            qty = self.get_position_qty(sym)
            price = self.get_current_price(sym)
            if price is not None:
                position_value = qty * price
                portfolio_value += position_value
                print(f"  {sym}: {qty:.6f} @ ${price:.2f} = ${position_value:.2f}")
                
        print(f"Total portfolio value: ${portfolio_value:.2f}")
        
        # Use TOTAL_CAPITAL directly for allocation
        total_allocation = TOTAL_CAPITAL
        print(f"Total allocation: ${total_allocation:.2f}")
        
        # Scale target allocations to fit within available balance
        scale_factor = min(1.0, available_balance / total_allocation)
        print(f"Scale factor: {scale_factor:.2f}")
        
        # Rebalance according to signals within available funds
        print("Rebalancing targets:")
        for sym, weight in pos_signals.items():
            target_notional = weight * total_allocation * scale_factor
            print(f"  {sym}: weight={weight:.2f}, target=${target_notional:.2f}")
            self.rebalance_symbol(sym, target_notional)
            # Update available balance after each order
            available_balance -= target_notional

    def rebalance_symbol(self, sym: str, target_notional: float):
        """
        Attempt to hold 'target_notional' USD worth of the slash-based symbol. 
        We'll get the last bar close as a 'current price' and compute desired qty.
        Then compare with current position (slash-based).
        """
        last_price = self.get_current_price(sym)
        if last_price is None:
            print(f"No price for {sym}, skip rebalance.")
            return

        desired_qty = 0
        if last_price > 0:
            desired_qty = target_notional / last_price

        current_qty = self.get_position_qty(sym)
        delta = desired_qty - current_qty

        # Minimum trade size in USD
        MIN_TRADE_SIZE = 10.0
        
        # Calculate notional delta
        notional_delta = round(delta * last_price, 2)
        
        # Skip if trade size is too small
        if abs(notional_delta) < MIN_TRADE_SIZE:
            print(f"{sym} trade size ${notional_delta:.2f} is below minimum ${MIN_TRADE_SIZE:.2f}, skip.")
            return

        if delta > 0:
            # buy
            print(f"[REB] Buying ~${notional_delta:.2f} of {sym}, delta qty={delta:.6f}")
            try:
                self.api.submit_order(
                    symbol=sym,
                    notional=notional_delta,
                    side='buy',
                    type='market',
                    time_in_force='gtc'
                )
            except Exception as e:
                print(f"Buy error for {sym}: {e}")
        else:
            # sell
            notional_delta = round(abs(delta) * last_price, 2)
            print(f"[REB] Selling ~${notional_delta:.2f} of {sym}, delta qty={delta:.6f}")
            try:
                self.api.submit_order(
                    symbol=sym,
                    notional=notional_delta,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
            except Exception as e:
                print(f"Sell error for {sym}: {e}")

    def get_current_price(self, sym: str) -> float:
        """
        We'll fetch the last 1-min bar with limit=1 for the slash-based symbol, 
        interpret the 'Close' as current price.
        """
        try:
            bars = self.api.get_crypto_bars(sym, tradeapi.TimeFrame.Minute, limit=1).df
            if bars.empty:
                return None
            # flatten multi-index if needed
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.reset_index(level=0, drop=True)
            # last row's close
            last_close = bars['close'].iloc[-1]
            return float(last_close)
        except Exception as e:
            print(f"Error fetching last bar price for {sym}: {e}")
            return None

    def get_position_qty(self, sym: str) -> float:
        """
        Attempt slash-based get_position. If no position, return 0.
        """
        try:
            pos = self.api.get_position(sym)
            return float(pos.qty)
        except tradeapi.rest.APIError as e:
            if "does not exist" in str(e).lower() or "404" in str(e):
                return 0.0
            else:
                print(f"Error fetching position for {sym}: {e}")
                return 0.0
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return 0.0
            else:
                print(f"HTTP Error fetching position for {sym}: {e}")
                return 0.0

    def calculate_portfolio_performance(self):
        """
        Calculate and display portfolio performance metrics
        """
        total_pnl = 0.0
        print("\nPortfolio Performance:")
        for sym, position in self.position_tracker.items():
            current_price = self.get_current_price(sym)
            if current_price is None:
                continue
                
            current_value = position['entry_qty'] * current_price
            pnl = current_value - position['total_cost']
            total_pnl += pnl
            
            print(f"  {sym}:")
            print(f"    Entry Price: ${position['entry_price']:.2f}")
            print(f"    Current Price: ${current_price:.2f}")
            print(f"    Quantity: {position['entry_qty']:.6f}")
            print(f"    PnL: ${pnl:.2f}")
            
        print(f"\nTotal PnL: ${total_pnl:.2f}")
        print(f"Total Fees: ${self.total_fees:.2f}")
        print(f"Net Profit: ${total_pnl - self.total_fees:.2f}")

    def liquidate_all(self):
        """
        Liquidate any of the slash-based symbols in CRYPTO_SYMBOLS if we hold them.
        """
        for sym in CRYPTO_SYMBOLS:
            qty = self.get_position_qty(sym)
            if qty > 0:
                last_price = self.get_current_price(sym)
                if last_price is None:
                    continue
                notional = qty * last_price
                print(f"[LIQUIDATE] Selling {qty:.6f} of {sym} ~${notional:.2f}.")
                try:
                    self.api.submit_order(
                        symbol=sym,
                        notional=notional,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    # Update position tracker
                    if sym in self.position_tracker:
                        del self.position_tracker[sym]
                except Exception as e:
                    print(f"Error liquidating {sym}: {e}")
        
        # Display final performance metrics
        self.calculate_portfolio_performance()

def main_loop():
    rebalancer = LiveRebalancer()
    while True:
        print(f"\n[{datetime.now()}] Starting rebalance cycle...")
        rebalancer.run_live_cycle()
        print(f"[{datetime.now()}] Rebalance done. Sleeping {REBALANCE_INTERVAL} seconds...\n")
        time.sleep(REBALANCE_INTERVAL)

if __name__ == "__main__":
    main_loop()
