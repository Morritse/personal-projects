import os
import json
from typing import Dict, Tuple
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# Alpaca credentials from fetch_working_data.py
API_KEY = "PKASZ48REAQARDXG66WF"
API_SECRET = "L8w2jmhDilnFSxFA9VLNMDbef0copxhf3NOTXSFH"

class OrderVerifier:
    def __init__(self):
        self.trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        self.data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
        
    def load_target_positions(self) -> Dict[str, float]:
        """Load target positions from positions.json"""
        positions_file = os.path.join(os.path.dirname(__file__), "positions.json")
        with open(positions_file, 'r') as f:
            return json.load(f)
    
    def get_current_positions(self) -> Dict[str, Tuple[int, float]]:
        """Get current positions as (shares, market_value)"""
        positions = self.trading_client.get_all_positions()
        
        current = {}
        for pos in positions:
            shares = int(pos.qty)
            if pos.side == 'short':
                shares = -shares
            current[pos.symbol] = (shares, float(pos.market_value))
            
        return current
    
    def get_open_orders(self) -> Dict[str, Tuple[int, str]]:
        """Get open orders as (shares, side)"""
        orders = self.trading_client.get_orders()
        
        open_orders = {}
        for order in orders:
            shares = int(order.qty)
            if order.side == 'sell':
                shares = -shares
            open_orders[order.symbol] = (shares, order.status)
            
        return open_orders
    
    def calculate_target_shares(self, symbol: str, multiplier: float, price: float, equity: float) -> int:
        """Calculate target number of shares based on position multiplier"""
        target_value = equity * multiplier
        return int(target_value / price)
    
    def verify_positions(self):
        """Compare current positions + open orders against targets"""
        print("\nVerifying positions and orders...")
        
        # Get account equity for target calculations
        account = self.trading_client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        print(f"\nAccount Details:")
        print(f"  Equity: ${equity:,.2f}")
        print(f"  Buying Power: ${buying_power:,.2f}")
        
        # Get current positions
        current_positions = self.get_current_positions()
        print("\nCurrent Positions:")
        for symbol, (shares, value) in current_positions.items():
            print(f"  {symbol}: {shares:,} shares (${value:,.2f})")
        
        # Get open orders
        open_orders = self.get_open_orders()
        if open_orders:
            print("\nOpen Orders:")
            for symbol, (shares, status) in open_orders.items():
                direction = "BUY" if shares > 0 else "SELL"
                print(f"  {symbol}: {direction} {abs(shares):,} shares ({status})")
        else:
            print("\nNo open orders")
        
        # Get target positions
        target_positions = self.load_target_positions()
        
        # Scale targets
        sum_abs = sum(abs(mult) for mult in target_positions.values())
        scaling_factor = min(1.95 / sum_abs, 1.0)
        scaled_targets = {s: m * scaling_factor for s, m in target_positions.items()}
        
        # Get current prices
        prices = {}
        for symbol in set(list(current_positions.keys()) + list(target_positions.keys())):
            try:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = self.data_client.get_stock_latest_quote(request)
                prices[symbol] = float(quote[symbol].ask_price)
            except Exception as e:
                print(f"Error getting quote for {symbol}: {str(e)}")
                return
        
        # Compare positions
        print("\nPosition Analysis:")
        print("=" * 80)
        
        total_current_value = 0
        total_target_value = 0
        total_pending_value = 0
        positions_completed = 0
        positions_in_progress = 0
        positions_not_started = 0
        
        for symbol in sorted(set(list(current_positions.keys()) + list(target_positions.keys()))):
            current_shares, current_value = current_positions.get(symbol, (0, 0.0))
            target_mult = scaled_targets.get(symbol, 0.0)
            target_shares = self.calculate_target_shares(symbol, target_mult, prices[symbol], equity)
            pending_shares = open_orders.get(symbol, (0, ''))[0]
            
            # Calculate values and signs
            target_value = target_shares * prices[symbol]
            pending_value = pending_shares * prices[symbol]
            
            # Update totals (use signed values)
            total_current_value += current_value
            total_target_value += target_value
            if pending_shares:
                total_pending_value += pending_value
            
            # Calculate expected final position
            expected_shares = current_shares + pending_shares
            
            # Track progress
            if expected_shares == target_shares:
                positions_completed += 1
            elif pending_shares != 0:
                positions_in_progress += 1
            else:
                positions_not_started += 1
            
            if expected_shares != target_shares:
                print(f"\n{symbol}:")
                # Format values with correct signs
                current_str = f"${current_value:>12,.2f}" if current_value >= 0 else f"(${abs(current_value):>11,.2f})"
                target_str = f"${target_value:>12,.2f}" if target_value >= 0 else f"(${abs(target_value):>11,.2f})"
                
                print(f"  Current:  {current_shares:>8,} shares {current_str}")
                if pending_shares:
                    direction = "+" if pending_shares > 0 else ""
                    pending_str = f"${pending_value:>12,.2f}" if pending_value >= 0 else f"(${abs(pending_value):>11,.2f})"
                    print(f"  Pending:  {direction}{pending_shares:>7,} shares {pending_str}")
                print(f"  Target:   {target_shares:>8,} shares {target_str}")
                print(f"  Expected: {expected_shares:>8,} shares")
                print(f"  Needed:   {(target_shares - expected_shares):>8,} shares")
                print("-" * 50)
        
        # Print summary
        print("\nProgress Summary:")
        print("=" * 80)
        print(f"Positions Completed:    {positions_completed:>3} of {len(scaled_targets):>3} ({positions_completed/len(scaled_targets)*100:>6.1f}%)")
        print(f"Positions In Progress:  {positions_in_progress:>3} of {len(scaled_targets):>3} ({positions_in_progress/len(scaled_targets)*100:>6.1f}%)")
        print(f"Positions Not Started:  {positions_not_started:>3} of {len(scaled_targets):>3} ({positions_not_started/len(scaled_targets)*100:>6.1f}%)")
        
        print("\nValue Summary:")
        print("=" * 80)
        # Format summary values with correct signs
        current_str = f"${total_current_value:>14,.2f}" if total_current_value >= 0 else f"(${abs(total_current_value):>13,.2f})"
        pending_str = f"${total_pending_value:>14,.2f}" if total_pending_value >= 0 else f"(${abs(total_pending_value):>13,.2f})"
        target_str = f"${total_target_value:>14,.2f}" if total_target_value >= 0 else f"(${abs(total_target_value):>13,.2f})"
        
        print(f"Current Position Value: {current_str}")
        print(f"Pending Changes Value:  {pending_str}")
        print(f"Target Position Value:  {target_str}")

def main():
    verifier = OrderVerifier()
    verifier.verify_positions()

if __name__ == "__main__":
    main()
