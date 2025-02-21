import time
import math
import json
from datetime import datetime
import alpaca_trade_api as tradeapi  # or 'alpaca-py', whichever you prefer

# 1) ALPACA CREDENTIALS + API
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env

# ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
# ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_API_KEY="PKB47VPLE9OYP875O68E"
ALPACA_SECRET_KEY="cVNZxQ7LbGlYu5zhqCMTkTVgKKlatOjM4BY3Y3Pv"
BASE_URL = "https://paper-api.alpaca.markets"

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env file")

# Initialize
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

TOTAL_CAPITAL_TO_DEPLOY = 0.9  # target 90% of account equity to avoid slippage issues
SIGNAL_THRESHOLD = 0.05        # ignore signals whose absolute value < this

def run_quant_and_get_signals():
    """
    Run quant.py to generate fresh signals and return them
    """
    try:
        print("\nGenerating fresh signals via quant.py...")
        os.system('python quant.py')
        
        with open("signals.json", "r") as f:
            signals = json.load(f)
            print(f"Generated new signals: {signals}")
            return signals
    except Exception as e:
        print(f"\nError running quant.py: {str(e)}")
        return {}

def get_positions_dict():
    """
    Returns a dict {symbol: (side, qty)} of current positions from Alpaca
    """
    positions = api.list_positions()
    pos_dict = {}
    for p in positions:
        sym = p.symbol
        qty = float(p.qty)
        side = 'long' if qty > 0 else 'short'
        pos_dict[sym] = (side, abs(qty))
    return pos_dict

def is_market_open():
    """Check if the market is currently open"""
    try:
        clock = api.get_clock()
        return clock.is_open
    except Exception as e:
        print(f"Error checking market status: {str(e)}")
        return False

def main_loop():
    print("\nTrader starting. Will run quant.py every minute to generate fresh signals.")
    print("Note: Trades will only execute during market hours.")
    
    while True:
        # 0) Check if market is open
        market_open = is_market_open()
        if not market_open:
            print("\nMarket is closed. Waiting 60s before checking again...")
            time.sleep(60)
            continue
            
        # 1) Generate fresh signals by running quant.py
        signals = run_quant_and_get_signals()
        if not signals:
            print("No valid signals generated. Waiting 60s before retry...")
            time.sleep(60)
            continue
            
        # 2) Fetch current equity from Alpaca
        account = api.get_account()
        equity = float(account.equity)  # total $ in the account
        # We'll only allocate a fraction of this:
        max_alloc = equity * TOTAL_CAPITAL_TO_DEPLOY
        
        # 3) Filter out near-zero signals
        filtered_signals = { s: v for s,v in signals.items() if abs(v) >= SIGNAL_THRESHOLD }
        if not filtered_signals:
            print("All signals below threshold. Possibly close all positions or do nothing.")
            # Optionally close everything
            time.sleep(60)
            continue
        
        # 4) Normalize signals to sum(abs(...))=1
        abs_sum = sum(abs(v) for v in filtered_signals.values())
        if abs_sum == 0:
            # If they're all zero or extremely small
            print("Sum of signals is 0, skip trading.")
            time.sleep(60)
            continue
        
        # Weighted signals in [-1..+1], sum of absolute weights = 1
        weights = { s: (v / abs_sum) for s,v in filtered_signals.items() }
        
        # 5) Determine the target dollar for each symbol
        # If weight is +0.2 => allocate 20% of max_alloc => go long
        # If weight is -0.2 => allocate 20% of max_alloc => go short
        target_dollars = {}
        for sym, w in weights.items():
            target_dollars[sym] = w * max_alloc  # can be negative for short
            
        # 6) Fetch current positions & see how far we are from target
        current_positions = get_positions_dict()
        
        # 7) First handle all sells to free up capital
        print("\nProcessing sell orders first...")
        for sym, tgt_val in target_dollars.items():
            last_quote = api.get_latest_trade(sym)
            price = float(last_quote.price) if last_quote and last_quote.price else None
            if not price:
                print(f" No price for {sym}, skip.")
                continue
            
            desired_shares = int(math.floor(tgt_val / price))
            current_shares = 0
            if sym in current_positions:
                side, qty = current_positions[sym]
                current_shares = qty if side=='long' else -qty
            
            diff_shares = desired_shares - current_shares
            
            if diff_shares < 0:
                # Need to sell - check actual available quantity
                available_qty = 0
                if sym in current_positions:
                    _, qty = current_positions[sym]
                    available_qty = qty
                
                # Use minimum of calculated quantity and available quantity
                sell_qty = min(abs(diff_shares), available_qty)
                if sell_qty > 0:
                    print(f"Selling {sell_qty} shares of {sym} (reducing position by {abs(diff_shares)} out of {available_qty} held)")
                    api.submit_order(
                        symbol=sym,
                        qty=sell_qty,
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
        
        # Get updated cash after sells
        account = api.get_account()
        cash = float(account.cash)
        print(f"\nAvailable cash after sells: ${cash:,.2f}")
        
        # 8) Then handle buys with available buying power
        print("\nProcessing buy orders with available buying power...")
        buy_orders = []
        total_buy_cost = 0
        
        # First calculate all desired buys
        for sym, tgt_val in target_dollars.items():
            last_quote = api.get_latest_trade(sym)
            price = float(last_quote.price) if last_quote and last_quote.price else None
            if not price:
                continue
                
            desired_shares = int(math.floor(tgt_val / price))
            current_shares = 0
            if sym in current_positions:
                side, qty = current_positions[sym]
                current_shares = qty if side=='long' else -qty
            
            diff_shares = desired_shares - current_shares
            
            if diff_shares > 0:
                cost = diff_shares * price
                buy_orders.append((sym, diff_shares, cost))
                total_buy_cost += cost
        
        # Scale orders to use 90% of available buying power
        if total_buy_cost > 0:
            print(f"\nTotal cost of desired buys: ${total_buy_cost:,.2f}")
            print(f"Available cash: ${cash:,.2f}")
            
            # Use 80% of cash to be safe
            target_cash = cash * 0.8
            scale = target_cash / total_buy_cost
            
            # Never scale up, only down
            if scale > 1.0:
                scale = 1.0
                
            print(f"Scaling orders to {scale:.1%} to target ${target_cash:,.2f}")
            
            # Scale all orders by the same factor
            buy_orders = [(sym, int(shares * scale), cost * scale) for sym, shares, cost in buy_orders]
            scaled_total = sum(cost for _, _, cost in buy_orders)
            print(f"Final total cost after scaling: ${scaled_total:,.2f}")
            
            # Execute buy orders while tracking remaining cash
            remaining_cash = target_cash
            print(f"\nStarting with ${remaining_cash:,.2f} cash")
            
            for sym, shares, cost in buy_orders:
                if shares > 0:
                    # Check if we can afford this order
                    if cost > remaining_cash:
                        # Scale down this order to fit remaining cash
                        scale = remaining_cash / cost
                        shares = int(shares * scale)
                        cost = shares * float(api.get_latest_trade(sym).price)
                        if shares == 0:
                            print(f"Skipping {sym} - insufficient cash")
                            continue
                    
                    print(f"Buying {shares} shares of {sym} (${cost:,.2f})")
                    api.submit_order(
                        symbol=sym,
                        qty=shares,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    # Update remaining cash
                    remaining_cash -= cost
                    print(f"Remaining cash: ${remaining_cash:,.2f}")
        
        # 8) Optionally, close out positions that no longer appear in signals
        # so if a symbol not in 'filtered_signals' but we hold shares, we might close it:
        for sym in current_positions:
            if sym not in filtered_signals:
                print(f"Closing out leftover symbol {sym}")
                side, qty = current_positions[sym]
                sell_side = 'sell' if side=='long' else 'buy'
                api.submit_order(
                    symbol=sym,
                    qty=qty,
                    side=sell_side,
                    type='market',
                    time_in_force='day'
                )
        
        # 9) Wait a minute
        print("Sleeping for 60s...")
        time.sleep(60)

if __name__ == "__main__":
    main_loop()
