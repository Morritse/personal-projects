"""
Place market-on-open orders for next trading day based on our trading plan.
"""

import os
import json
from datetime import datetime
import alpaca_trade_api as tradeapi
from market_analyzer import fetch_market_data, MarketData

# Initialize Alpaca paper trading API
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(
    key_id=ALPACA_API_KEY,
    secret_key=ALPACA_SECRET_KEY,
    base_url=ALPACA_PAPER_URL,
    api_version='v2'
)

def load_trading_plan():
    """Load and validate trading plan"""
    with open('trading_plan.json', 'r') as f:
        return json.load(f)

def calculate_stop_price(stop_expr: str, current_price: float) -> float:
    """Calculate stop price based on expression and current price"""
    try:
        if isinstance(stop_expr, (int, float)):
            return float(stop_expr)
        elif 'market *' in stop_expr:
            # Handle percentage-based stops
            multiplier = float(stop_expr.split('*')[1].strip())
            return round(current_price * multiplier, 2)
        else:
            # Default to direct price
            return round(float(stop_expr), 2)
    except Exception as e:
        print(f"Error calculating stop price: {str(e)}")
        return round(current_price * 0.90, 2)  # Default to 10% stop

def prepare_orders(trading_plan, portfolio_value: float):
    """Prepare orders based on trading plan"""
    orders = []
    
    # Calculate total planned allocation
    total_allocation = sum(
        sum(float(zone['size'].strip('%')) for zone in pos['entry']['entry_zones'])
        for pos in trading_plan['trading_plan']['positions']
    )
    
    # Calculate scaling factor to hit 90% target
    scale = 90.0 / total_allocation
    
    # Prepare orders for each position
    for position in trading_plan['trading_plan']['positions']:
        symbol = position['symbol']
        
        try:
            # Get current market data
            data = MarketData.from_json(fetch_market_data(symbol))
            current_price = data.close
            
            # Calculate total position size
            total_size_pct = sum(float(zone['size'].strip('%')) for zone in position['entry']['entry_zones'])
            scaled_size_pct = min(total_size_pct * scale, 20.0)  # Cap at 20%
            position_value = portfolio_value * (scaled_size_pct / 100)
            # Round shares to whole numbers
            shares = round(position_value / current_price)
            
            # Skip if less than 1 share
            if shares < 1:
                continue

            # Calculate stop loss price
            stop_price = calculate_stop_price(position['entry']['stop_loss']['price'], current_price)

            # Single entry order for total position
            orders.append({
                'type': 'entry',
                'symbol': symbol,
                'qty': shares,
                'side': 'buy',
                'order_type': 'market',
                'time_in_force': 'day',
                'targets': [],  # No targets - will reevaluate daily
                'stop_loss': {
                    'price': stop_price
                }
            })
        
        except Exception as e:
            print(f"Error preparing orders for {symbol}: {str(e)}")
    
    return orders

def place_entry_orders():
    """Place entry orders for next market open"""
    try:
        # Load trading plan
        trading_plan = load_trading_plan()
        
        # Get account info
        account = api.get_account()
        portfolio_value = float(account.portfolio_value)
        
        print(f"\nPaper Account Status:")
        print(f"Account ID: {account.id}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Target Allocation: 90% (${portfolio_value * 0.9:,.2f})")
        
        # Prepare orders based on trading plan
        orders = prepare_orders(trading_plan, portfolio_value)
        
        # Get current positions
        current_positions = {
            p.symbol: {
                'qty': float(p.qty),
                'market_value': float(p.market_value)
            }
            for p in api.list_positions()
        }
        
        # Cancel existing orders (but keep positions)
        api.cancel_all_orders()
        print("\nCanceled existing orders")
        
        # Analyze position changes needed
        position_changes = {}
        for order in orders:
            symbol = order['symbol']
            target_shares = order['qty']
            
            if symbol in current_positions:
                current_shares = current_positions[symbol]['qty']
                diff = target_shares - current_shares
                
                if abs(diff) >= 1:  # Only trade if difference is 1 share or more
                    position_changes[symbol] = {
                        'action': 'adjust',
                        'shares': diff,
                        'targets': order['targets'],
                        'stop_loss': order['stop_loss']
                    }
                else:
                    print(f"\nKeeping {symbol} position unchanged:")
                    print(f"Current shares: {current_shares}")
            else:
                position_changes[symbol] = {
                    'action': 'new',
                    'shares': target_shares,
                    'targets': order['targets'],
                    'stop_loss': order['stop_loss']
                }
        
        # Close positions not in trading plan
        for symbol in current_positions:
            if symbol not in [o['symbol'] for o in orders]:
                print(f"\nClosing {symbol} - not in trading plan")
                api.close_position(symbol)
        
        # Execute position changes
        if position_changes:
            print("\nExecuting position changes:")
            for symbol, change in position_changes.items():
                try:
                    if change['action'] == 'new':
                        # New position - place market order
                        entry = api.submit_order(
                            symbol=symbol,
                            qty=abs(change['shares']),
                            side='buy',
                            type='market',
                            time_in_force='day'
                        )
                        print(f"\nNew position order for {symbol}:")
                        print(f"Shares: {change['shares']} @ Market")
                        print(f"Stop Loss: ${change['stop_loss']['price']:.2f}")
                    
                    elif change['action'] == 'adjust':
                        # Adjust existing position
                        side = 'buy' if change['shares'] > 0 else 'sell'
                        entry = api.submit_order(
                            symbol=symbol,
                            qty=abs(change['shares']),
                            side=side,
                            type='market',
                            time_in_force='day'
                        )
                        print(f"\nAdjusting {symbol} position:")
                        print(f"{side.title()}: {abs(change['shares'])} shares @ Market")
                
                except Exception as e:
                    print(f"Error submitting order for {symbol}: {str(e)}")
            
            # Show unique pending orders
            seen = set()
            print("\nPending Entry Orders:")
            for order in api.list_orders(status='open'):
                if order.symbol not in seen:
                    print(f"{order.symbol}: {order.qty} shares @ Market")
                    seen.add(order.symbol)
        else:
            print("\nNo orders to place for tomorrow")
    
    except Exception as e:
        print(f"Error setting up trades: {str(e)}")

def place_exit_orders():
    """Place stop loss and take profit orders for filled entries"""
    try:
        # Load trading plan
        trading_plan = load_trading_plan()
        plan_by_symbol = {pos['symbol']: pos for pos in trading_plan['trading_plan']['positions']}
        
        # Get filled positions
        positions = api.list_positions()
        
        for position in positions:
            symbol = position.symbol
            if symbol in plan_by_symbol:
                plan = plan_by_symbol[symbol]
                shares = float(position.qty)
                current_price = float(position.current_price)
                
                # Calculate and place stop loss order
                stop_price = calculate_stop_price(plan['entry']['stop_loss']['price'], current_price)
                api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='sell',
                    type='stop_limit',
                    time_in_force='gtc',
                    stop_price=stop_price,
                    limit_price=round(stop_price * 0.99, 2)  # 1% buffer
                )
                print(f"\nStop Loss placed for {symbol}:")
                print(f"Shares: {shares} @ ${stop_price:.2f}")
                
                # No take profit orders - will reevaluate daily
    
    except Exception as e:
        print(f"Error placing exit orders: {str(e)}")

def wait_for_fills(symbols, timeout_minutes=5):
    """Wait for entry orders to fill"""
    import time
    start_time = time.time()
    timeout = timeout_minutes * 60  # convert to seconds
    
    while time.time() - start_time < timeout:
        # Get current positions
        positions = {p.symbol: p for p in api.list_positions()}
        
        # Check if all symbols are filled
        all_filled = True
        for symbol in symbols:
            if symbol not in positions:
                all_filled = False
                break
        
        if all_filled:
            print("\nAll entry orders filled!")
            return True
        
        # Wait 30 seconds before checking again
        print(f"\nWaiting for entries to fill... ({timeout_minutes} minute timeout)")
        time.sleep(30)
    
    return False

def main():
    """Place orders after market close for next day's open"""
    # Get market hours
    clock = api.get_clock()
    
    if not clock.is_open:
        print("\nMarket closed. Setting up orders for tomorrow's open...")
        place_entry_orders()
        
        print("\nNOTE: Exit orders will be placed automatically after entries fill")
        print("      at market open tomorrow. The script will:")
        print("      1. Wait for market to open")
        print("      2. Wait up to 5 minutes for entries to fill")
        print("      3. Place stop-loss and take-profit orders")
        
    else:
        # During market hours - check for fills and place exits
        print("\nMarket is open. Checking for filled entries...")
        
        # Get symbols from trading plan
        trading_plan = load_trading_plan()
        symbols = [pos['symbol'] for pos in trading_plan['trading_plan']['positions']]
        
        # Wait for fills
        if wait_for_fills(symbols):
            print("\nPlacing exit orders for filled positions...")
            place_exit_orders()
        else:
            print("\nTimeout waiting for fills. Please check positions manually")
            print("and run with --exits-only if needed.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--exits-only':
        # Only place exit orders for filled positions
        place_exit_orders()
    else:
        # Normal operation
        main()
