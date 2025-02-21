import os
import json
import time
from typing import Dict, Any
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Position
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest

# Alpaca credentials from fetch_working_data.py
API_KEY = "PKASZ48REAQARDXG66WF"
API_SECRET = "L8w2jmhDilnFSxFA9VLNMDbef0copxhf3NOTXSFH"

class PortfolioRebalancer:
    def __init__(self):
        self.trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
        self.data_client = StockHistoricalDataClient(API_KEY, API_SECRET)
        
    def load_target_positions(self) -> Dict[str, float]:
        """Load target positions from positions.json"""
        positions_file = os.path.join(os.path.dirname(__file__), "positions.json")
        with open(positions_file, 'r') as f:
            return json.load(f)
    
    def get_current_positions(self) -> Dict[str, int]:
        """
        Get current positions as actual share counts.
        Positive numbers for long positions, negative for shorts.
        """
        # Get current positions
        positions = self.trading_client.get_all_positions()
        
        # Get share counts
        current_positions = {}
        for pos in positions:
            shares = int(pos.qty)
            # Make negative if short
            if pos.side == 'short':
                shares = -shares
            current_positions[pos.symbol] = shares
            
        return current_positions
    
    def scale_positions(self, positions: Dict[str, float], margin: float = 1.95) -> Dict[str, float]:
        """
        Scale positions if sum of absolute values exceeds margin limit.
        
        Args:
            positions: Dictionary of position multipliers
            margin: Maximum allowed sum of absolute positions (e.g. 2.0 for 2:1 margin)
        
        Returns:
            Dictionary of scaled position multipliers
        """
        # Calculate sum of absolute positions
        sum_abs = sum(abs(mult) for mult in positions.values())
        
        # If sum is within margin limit, no scaling needed
        if sum_abs <= margin:
            return positions.copy()
        
        # Calculate scaling factor
        scaling_factor = margin / sum_abs
        
        # Apply scaling
        return {symbol: mult * scaling_factor for symbol, mult in positions.items()}

    def calculate_target_shares(self, symbol: str, multiplier: float, price: float, equity: float) -> int:
        """Calculate target number of shares based on position multiplier"""
        target_value = equity * multiplier
        return int(target_value / price)
    
    def calculate_trades(self, current: Dict[str, int], target: Dict[str, float]) -> Dict[str, int]:
        """Calculate required share changes"""
        trades = {}
        
        # Get account equity and scale targets
        account = self.trading_client.get_account()
        equity = float(account.equity)
        scaled_targets = self.scale_positions(target)
        
        # Get current prices
        prices = {}
        all_symbols = set(list(current.keys()) + list(scaled_targets.keys()))
        for symbol in all_symbols:
            try:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = self.data_client.get_stock_latest_quote(request)
                prices[symbol] = float(quote[symbol].ask_price)
            except Exception as e:
                print(f"Error getting quote for {symbol}: {str(e)}")
                # If there's an error, skip the symbol
                continue
        
        # Calculate required share changes
        for symbol, target_mult in scaled_targets.items():
            if symbol not in prices:
                continue
            current_shares = current.get(symbol, 0)
            target_shares = self.calculate_target_shares(symbol, target_mult, prices[symbol], equity)
            
            # Calculate share difference
            delta_shares = target_shares - current_shares
            
            # Only trade if change is non-zero
            if abs(delta_shares) > 0:
                trades[symbol] = delta_shares
        
        return trades
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            print(f"Error checking market hours: {str(e)}")
            return False
    
    def execute_order(self, symbol: str, shares: int, price: float) -> bool:
        """Execute a single order (market) and do a limited check of status."""
        try:
            if shares == 0:
                return True
            
            # Determine order side
            side = OrderSide.BUY if shares > 0 else OrderSide.SELL
            
            print(f"\nPlacing {side.value.upper()} order for {symbol}: {abs(shares)} shares at ~${price:.2f}")
            
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=abs(shares),  # Use absolute value for quantity
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            print(f"  Order submitted. ID = {order.id}, status = {order.status}")
            
            # If the market is open, we can briefly wait for fill
            if self.is_market_open():
                # Wait up to ~15 seconds for a fill (in practice can be shorter/longer)
                start_time = time.time()
                max_wait = 15
                while time.time() - start_time < max_wait:
                    refreshed_order = self.trading_client.get_order_by_id(order.id)
                    if refreshed_order.status in ['filled', 'partially_filled']:
                        print(f"  -> Order filled with status: {refreshed_order.status}")
                        return True
                    elif refreshed_order.status in ['canceled', 'expired', 'rejected']:
                        print(f"  -> Order {refreshed_order.status}")
                        return False
                    time.sleep(2)
                print("  -> Timed out waiting for fill. Check later.")
                return True
            else:
                # After hours, the order won't fill now.
                # We'll rely on it filling or canceling next market open.
                print("  Market is closed. Order accepted for next session.")
                return True
            
        except Exception as e:
            print(f"Error trading {symbol}: {str(e)}")
            return False
    
    def execute_trades(self, trades: Dict[str, int], current_positions: Dict[str, int]):
        """Execute the calculated trades without re-checking 'closing, then re-opening' in a loop."""
        account = self.trading_client.get_account()
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        
        print(f"\nAccount Details:")
        print(f"  Equity: ${equity:,.2f}")
        print(f"  Buying Power: ${buying_power:,.2f}")
        
        # Grab quotes
        prices = {}
        for symbol in trades:
            try:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote = self.data_client.get_stock_latest_quote(request)
                prices[symbol] = float(quote[symbol].ask_price)
            except Exception as e:
                print(f"Error getting quote for {symbol}: {str(e)}")
                prices[symbol] = 0.0
        
        # Just place the net trade for each symbol. 
        # If we need to reduce a position first, we place a SELL. 
        # If we need to open/increase, we place a BUY.
        print("\nExecuting trades (net changes)...")
        for symbol, delta_shares in trades.items():
            if symbol not in prices or prices[symbol] <= 0:
                print(f"  Skipping {symbol} (no valid price).")
                continue
            
            # Net trade = target_shares - current_shares
            # e.g. if we have 100 shares, target=150, delta=50 => buy 50
            # if we have 100, target=20, delta=-80 => sell 80
            self.execute_order(symbol, delta_shares, prices[symbol])
    
    def rebalance(self):
        """Main rebalancing process"""
        print("\nStarting portfolio rebalance...")
        
        # 1. Load target positions
        try:
            target_positions = self.load_target_positions()
            print("\nRaw target positions:")
            for symbol, target in target_positions.items():
                print(f"  {symbol}: {target:.2f}")
        except Exception as e:
            print(f"Error loading target positions: {str(e)}")
            return
        
        # 2. Get current positions
        try:
            current_positions = self.get_current_positions()
            print("\nCurrent positions:")
            for symbol, pos in current_positions.items():
                print(f"  {symbol}: {pos}")
        except Exception as e:
            print(f"Error getting current positions: {str(e)}")
            return
        
        # 3. Calculate required trades
        trades = self.calculate_trades(current_positions, target_positions)
        if not trades:
            print("\nNo significant changes required.")
            return
        
        print("\nRequired trades:")
        for symbol, delta in trades.items():
            direction = "BUY" if delta > 0 else "SELL"
            print(f"  {symbol}: {direction} {abs(delta)} shares")
        
        # 4. Execute trades
        self.execute_trades(trades, current_positions)
        
        print("\nRebalancing complete!")

def main():
    rebalancer = PortfolioRebalancer()
    rebalancer.rebalance()

if __name__ == "__main__":
    main()
