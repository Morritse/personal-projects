import os
import time
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Replace with your Alpaca API key and secret
API_KEY = "PKASZ48REAQARDXG66WF"
API_SECRET = "L8w2jmhDilnFSxFA9VLNMDbef0copxhf3NOTXSFH"
PAPER = True  # Change to False if using a live account (WARNING: real trades!)

def wait_for_order_fill(trading_client, order, max_wait=60):
    """Wait for an order to fill, with timeout"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            current = trading_client.get_order(order.id)
            if current.status == "filled":
                print(f"  -> Order filled")
                return True
            elif current.status == "partially_filled":
                print(f"  -> Order partially filled ({current.filled_qty} shares)")
                return True
            elif current.status in ["canceled", "expired", "rejected", "suspended"]:
                print(f"  -> Order {current.status}")
                return False
            time.sleep(2)
        except Exception as e:
            print(f"  Error checking order status: {e}")
            return False
    print("  -> Order timed out")
    return False

def close_position_in_chunks(trading_client, position, chunk_size=25, max_retries=3):
    """Close a single position in smaller chunks to manage buying power requirements"""
    symbol = position.symbol
    qty = abs(float(position.qty))
    remaining = qty
    
    print(f"\nClosing {position.symbol} position:")
    print(f"  Side: {position.side}")
    print(f"  Quantity: {qty}")
    print(f"  Market Value: ${float(position.market_value):,.2f}")
    
    while remaining > 0:
        # Calculate chunk size for this iteration
        current_chunk = min(remaining, chunk_size)
        
        # Try to close this chunk
        for attempt in range(max_retries):
            try:
                print(f"  Closing {current_chunk} of {remaining} remaining shares...")
                
                # Create market order to close position
                side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=current_chunk,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    reduce_only=True  # Only reduce existing position
                )
                
                # Submit order and wait for fill
                order = trading_client.submit_order(order_request)
                print(f"  Order submitted. ID = {order.id}, status = {order.status}")
                
                # Wait for fill
                if wait_for_order_fill(trading_client, order):
                    # Update remaining shares
                    remaining -= current_chunk
                    if remaining > 0:
                        print(f"  {remaining} shares remaining")
                        # Get updated position
                        try:
                            position = trading_client.get_position(symbol)
                            print(f"  Current position: {position.qty} shares")
                        except Exception:
                            print("  Could not get updated position")
                        # Add delay between chunks
                        time.sleep(2)
                    break
                else:
                    # If order didn't fill, try again with smaller size
                    if attempt < max_retries - 1:
                        current_chunk = max(1, current_chunk // 2)
                        print(f"  Reducing chunk size to {current_chunk} shares")
                        time.sleep(2)
                    else:
                        print(f"  Failed to close chunk after {max_retries} attempts")
                        return False
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # If we hit buying power issues, reduce chunk size
                    if "buying power" in str(e).lower():
                        current_chunk = max(1, current_chunk // 2)
                        print(f"  Reducing chunk size to {current_chunk} shares")
                    else:
                        print(f"  Error closing {symbol}, retrying: {e}")
                    time.sleep(2)
                else:
                    print(f"  Failed to close {symbol} after {max_retries} attempts: {e}")
                    return False
    
    return True

def close_all_positions():
    # Initialize Alpaca trading client
    trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)
    
    print("Cancelling all open orders...")
    
    # Cancel existing orders
    print("  Getting open orders...")
    try:
        orders = trading_client.get_orders()
        if orders:
            print(f"  Found {len(orders)} orders")
            for order in orders:
                try:
                    trading_client.cancel_order(order_id=order.id)
                    print(f"  Cancelled order {order.id} for {order.symbol}")
                except Exception as e:
                    print(f"  Could not cancel {order.id}: {e}")
            print("  Waiting for cancellations to process...")
            time.sleep(5)
        else:
            print("  No open orders found")
    except Exception as e:
        print(f"  Error getting orders: {e}")
        return
    
    print("\nGetting current positions...")
    try:
        # Get all positions
        positions = trading_client.get_all_positions()
        
        if not positions:
            print("No open positions to close")
            return
        
        # Calculate total value
        total_value = sum(abs(float(pos.market_value)) for pos in positions)
        print(f"\nTotal position value: ${total_value:,.2f}")
        print(f"Number of positions: {len(positions)}")
        
        # First close long positions
        print("\nClosing long positions...")
        long_positions = [pos for pos in positions if pos.side == "long"]
        for i, pos in enumerate(long_positions, 1):
            print(f"\nProgress: {i}/{len(long_positions)} long positions")
            if not close_position_in_chunks(trading_client, pos):
                print(f"Warning: Failed to fully close {pos.symbol}")
            # Add delay between positions
            time.sleep(2)
        
        # Then close short positions
        print("\nClosing short positions...")
        short_positions = [pos for pos in positions if pos.side == "short"]
        for i, pos in enumerate(short_positions, 1):
            print(f"\nProgress: {i}/{len(short_positions)} short positions")
            if not close_position_in_chunks(trading_client, pos):
                print(f"Warning: Failed to fully close {pos.symbol}")
            # Add delay between positions
            time.sleep(2)
        
        # Wait for final orders to process
        time.sleep(5)
        
        # Verify all positions are closed
        final_positions = trading_client.get_all_positions()
        if not final_positions:
            print("\nAll positions successfully closed!")
        else:
            print(f"\nWarning: {len(final_positions)} positions remain open:")
            for pos in final_positions:
                print(f"  {pos.symbol}: {pos.qty} shares (${float(pos.market_value):,.2f})")
        
    except Exception as e:
        print(f"Error closing positions: {e}")

def main():
    close_all_positions()

if __name__ == "__main__":
    main()
