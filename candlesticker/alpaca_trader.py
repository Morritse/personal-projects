"""
Execute trades on Alpaca paper trading account based on our analysis.
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

class AlpacaTrader:
    def __init__(self, trading_plan_file: str, target_allocation_pct: float = 90.0, max_position_pct: float = 20.0):
        """Initialize trader with trading plan and allocation settings"""
        # Load trading plan
        with open(trading_plan_file, 'r') as f:
            self.trading_plan = json.load(f)
        
        self.target_allocation_pct = target_allocation_pct
        self.max_position_pct = max_position_pct
        
        # Get account info
        self.account = api.get_account()
        portfolio_value = float(self.account.portfolio_value)
        
        print(f"\nPaper Account Status:")
        print(f"Account ID: {self.account.id}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Target Allocation: {target_allocation_pct}% (${portfolio_value * target_allocation_pct/100:,.2f})")
        print(f"Max Position Size: {max_position_pct}%")
        
        # Get original position sizes from plan
        self.original_sizes = {}
        for pos in self.trading_plan['trading_plan']['positions']:
            symbol = pos['symbol']
            size = sum(float(zone['size'].strip('%')) for zone in pos['entry']['entry_zones'])
            self.original_sizes[symbol] = size
        
        # Calculate scaling factor while respecting max position size
        total_plan_pct = sum(self.original_sizes.values())
        base_scale = target_allocation_pct / total_plan_pct
        
        # Check if any position would exceed max size
        max_original = max(self.original_sizes.values())
        max_scaled = max_original * base_scale
        
        if max_scaled > max_position_pct:
            # Scale down to respect max position size
            self.position_scale = max_position_pct / max_original
        else:
            self.position_scale = base_scale
        
        # Calculate and display final allocations
        print("\nPlanned Allocations:")
        for symbol, original_pct in self.original_sizes.items():
            final_pct = min(original_pct * self.position_scale, max_position_pct)
            print(f"{symbol}: {original_pct:.1f}% → {final_pct:.1f}%")
        
        total_allocation = sum(min(pct * self.position_scale, max_position_pct) 
                             for pct in self.original_sizes.values())
        print(f"\nTotal Planned Allocation: {total_allocation:.1f}%")
        
        # Track orders and positions
        self.orders = {}
        self.positions = {}
        self._load_existing_positions()
    
    def _load_existing_positions(self):
        """Load existing positions from Alpaca"""
        try:
            positions = api.list_positions()
            for position in positions:
                self.positions[position.symbol] = {
                    'qty': float(position.qty),
                    'entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc)
                }
            
            if positions:
                print("\nExisting Positions:")
                for symbol, pos in self.positions.items():
                    print(f"{symbol}: {pos['qty']} shares @ ${pos['entry_price']:.2f}")
        
        except Exception as e:
            print(f"Error loading positions: {str(e)}")
    
    def check_signals(self):
        """Check trading plan for entry/exit signals"""
        signals = []
        
        # Check each position in trading plan
        for position in self.trading_plan['trading_plan']['positions']:
            symbol = position['symbol']
            
            try:
                # Get current market data
                data = MarketData.from_json(fetch_market_data(symbol))
                current_price = data.close
                
                # If we don't have a position, check entry conditions
                if symbol not in self.positions:
                    for entry_zone in position['entry']['entry_zones']:
                        price_range = entry_zone['price'].split('-')
                        zone_low = float(price_range[0])
                        zone_high = float(price_range[1])
                        
                        if zone_low <= current_price <= zone_high:
                            # Calculate position size with cap
                            base_size_pct = float(entry_zone['size'].strip('%')) * self.position_scale / 100
                            size_pct = min(base_size_pct, self.max_position_pct / 100)
                            position_value = float(self.account.portfolio_value) * size_pct
                            shares = position_value / current_price
                            
                            signals.append({
                                'type': 'entry',
                                'symbol': symbol,
                                'side': 'buy',
                                'qty': shares,
                                'stop_loss': position['entry']['stop_loss']['price'],
                                'targets': position['entry']['targets']
                            })
                
                # If we have a position, check exit conditions
                elif symbol in self.positions:
                    position_data = self.positions[symbol]
                    
                    # Check stop loss
                    if current_price <= float(position['entry']['stop_loss']['price']):
                        signals.append({
                            'type': 'exit',
                            'symbol': symbol,
                            'side': 'sell',
                            'qty': position_data['qty'],
                            'reason': 'stop_loss'
                        })
                    
                    # Check take profit levels
                    for target in position['entry']['targets']:
                        if current_price >= float(target['price']):
                            # Calculate shares to sell at this target
                            target_pct = float(target['size'].strip('%')) / 100
                            shares_to_sell = position_data['qty'] * target_pct
                            
                            signals.append({
                                'type': 'exit',
                                'symbol': symbol,
                                'side': 'sell',
                                'qty': shares_to_sell,
                                'reason': 'take_profit'
                            })
            
            except Exception as e:
                print(f"Error checking signals for {symbol}: {str(e)}")
        
        return signals
    
    def execute_signals(self, signals: list):
        """Execute trading signals on Alpaca paper account"""
        for signal in signals:
            try:
                if signal['type'] == 'entry':
                    # Calculate take profit prices and quantities
                    take_profits = []
                    remaining_shares = signal['qty']
                    
                    for target in signal['targets']:
                        target_shares = signal['qty'] * (float(target['size'].strip('%')) / 100)
                        take_profits.append({
                            'limit_price': target['price'],
                            'qty': target_shares
                        })
                        remaining_shares -= target_shares
                    
                    # Place bracket order with multiple take profits
                    for i, tp in enumerate(take_profits):
                        # For each target, place a bracket order
                        order = api.submit_order(
                            symbol=signal['symbol'],
                            qty=tp['qty'],
                            side='buy',
                            type='market',
                            time_in_force='opg',  # Market-on-Open order
"""
Execute trades on Alpaca paper trading account based on our analysis.
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

class AlpacaTrader:
    def __init__(self, trading_plan_file: str, target_allocation_pct: float = 90.0, max_position_pct: float = 20.0):
        """Initialize trader with trading plan and allocation settings"""
        # Load trading plan
        with open(trading_plan_file, 'r') as f:
            self.trading_plan = json.load(f)
        
        self.target_allocation_pct = target_allocation_pct
        self.max_position_pct = max_position_pct
        
        # Get account info
        self.account = api.get_account()
        portfolio_value = float(self.account.portfolio_value)
        
        print(f"\nPaper Account Status:")
        print(f"Account ID: {self.account.id}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Target Allocation: {target_allocation_pct}% (${portfolio_value * target_allocation_pct/100:,.2f})")
        print(f"Max Position Size: {max_position_pct}%")
        
        # Get original position sizes from plan
        self.original_sizes = {}
        for pos in self.trading_plan['trading_plan']['positions']:
            symbol = pos['symbol']
            size = sum(float(zone['size'].strip('%')) for zone in pos['entry']['entry_zones'])
            self.original_sizes[symbol] = size
        
        # Calculate scaling factor while respecting max position size
        total_plan_pct = sum(self.original_sizes.values())
        base_scale = target_allocation_pct / total_plan_pct
        
        # Check if any position would exceed max size
        max_original = max(self.original_sizes.values())
        max_scaled = max_original * base_scale
        
        if max_scaled > max_position_pct:
            # Scale down to respect max position size
            self.position_scale = max_position_pct / max_original
        else:
            self.position_scale = base_scale
        
        # Calculate and display final allocations
        print("\nPlanned Allocations:")
        for symbol, original_pct in self.original_sizes.items():
            final_pct = min(original_pct * self.position_scale, max_position_pct)
            print(f"{symbol}: {original_pct:.1f}% → {final_pct:.1f}%")
        
        total_allocation = sum(min(pct * self.position_scale, max_position_pct) 
                             for pct in self.original_sizes.values())
        print(f"\nTotal Planned Allocation: {total_allocation:.1f}%")
        
        # Track orders and positions
        self.orders = {}
        self.positions = {}
        self._load_existing_positions()
    
    def _load_existing_positions(self):
        """Load existing positions from Alpaca"""
        try:
            positions = api.list_positions()
            for position in positions:
                self.positions[position.symbol] = {
                    'qty': float(position.qty),
                    'entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc)
                }
            
            if positions:
                print("\nExisting Positions:")
                for symbol, pos in self.positions.items():
                    print(f"{symbol}: {pos['qty']} shares @ ${pos['entry_price']:.2f}")
        
        except Exception as e:
            print(f"Error loading positions: {str(e)}")
    
    def check_signals(self):
        """Check trading plan for entry/exit signals"""
        signals = []
        
        # Check each position in trading plan
        for position in self.trading_plan['trading_plan']['positions']:
            symbol = position['symbol']
            
            try:
                # Get current market data
                data = MarketData.from_json(fetch_market_data(symbol))
                current_price = data.close
                
                # If we don't have a position, check entry conditions
                if symbol not in self.positions:
                    for entry_zone in position['entry']['entry_zones']:
                        price_range = entry_zone['price'].split('-')
                        zone_low = float(price_range[0])
                        zone_high = float(price_range[1])
                        
                        if zone_low <= current_price <= zone_high:
                            # Calculate position size with cap
                            base_size_pct = float(entry_zone['size'].strip('%')) * self.position_scale / 100
                            size_pct = min(base_size_pct, self.max_position_pct / 100)
                            position_value = float(self.account.portfolio_value) * size_pct
                            shares = position_value / current_price
                            
                            signals.append({
                                'type': 'entry',
                                'symbol': symbol,
                                'side': 'buy',
                                'qty': shares,
                                'stop_loss': position['entry']['stop_loss']['price'],
                                'targets': position['entry']['targets']
                            })
                
                # If we have a position, check exit conditions
                elif symbol in self.positions:
                    position_data = self.positions[symbol]
                    
                    # Check stop loss
                    if current_price <= float(position['entry']['stop_loss']['price']):
                        signals.append({
                            'type': 'exit',
                            'symbol': symbol,
                            'side': 'sell',
                            'qty': position_data['qty'],
                            'reason': 'stop_loss'
                        })
                    
                    # Check take profit levels
                    for target in position['entry']['targets']:
                        if current_price >= float(target['price']):
                            # Calculate shares to sell at this target
                            target_pct = float(target['size'].strip('%')) / 100
                            shares_to_sell = position_data['qty'] * target_pct
                            
                            signals.append({
                                'type': 'exit',
                                'symbol': symbol,
                                'side': 'sell',
                                'qty': shares_to_sell,
                                'reason': 'take_profit'
                            })
            
            except Exception as e:
                print(f"Error checking signals for {symbol}: {str(e)}")
        
        return signals
    
    def execute_signals(self, signals: list):
        """Execute trading signals on Alpaca paper account"""
        for signal in signals:
            try:
                if signal['type'] == 'entry':
                    # Calculate take profit prices and quantities
                    take_profits = []
                    remaining_shares = signal['qty']
                    
                    for target in signal['targets']:
                        target_shares = signal['qty'] * (float(target['size'].strip('%')) / 100)
                        take_profits.append({
                            'limit_price': target['price'],
                            'qty': target_shares
                        })
                        remaining_shares -= target_shares
                    
                    # Place bracket order with multiple take profits
                    for i, tp in enumerate(take_profits):
                        # For each target, place a bracket order
                        order = api.submit_order(
                            symbol=signal['symbol'],
                            qty=tp['qty'],
                            side='buy',

"""
Execute trades on Alpaca paper trading account based on our analysis.
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

class AlpacaTrader:
    def __init__(self, trading_plan_file: str, target_allocation_pct: float = 90.0, max_position_pct: float = 20.0):
        """Initialize trader with trading plan and allocation settings"""
        # Load trading plan
        with open(trading_plan_file, 'r') as f:
            self.trading_plan = json.load(f)
        
        self.target_allocation_pct = target_allocation_pct
        self.max_position_pct = max_position_pct
        
        # Get account info
        self.account = api.get_account()
        portfolio_value = float(self.account.portfolio_value)
        
        print(f"\nPaper Account Status:")
        print(f"Account ID: {self.account.id}")
        print(f"Portfolio Value: ${portfolio_value:,.2f}")
        print(f"Target Allocation: {target_allocation_pct}% (${portfolio_value * target_allocation_pct/100:,.2f})")
        print(f"Max Position Size: {max_position_pct}%")
        
        # Get original position sizes from plan
        self.original_sizes = {}
        for pos in self.trading_plan['trading_plan']['positions']:
            symbol = pos['symbol']
            size = sum(float(zone['size'].strip('%')) for zone in pos['entry']['entry_zones'])
            self.original_sizes[symbol] = size
        
        # Calculate scaling factor while respecting max position size
        total_plan_pct = sum(self.original_sizes.values())
        base_scale = target_allocation_pct / total_plan_pct
        
        # Check if any position would exceed max size
        max_original = max(self.original_sizes.values())
        max_scaled = max_original * base_scale
        
        if max_scaled > max_position_pct:
            # Scale down to respect max position size
            self.position_scale = max_position_pct / max_original
        else:
            self.position_scale = base_scale
        
        # Calculate and display final allocations
        print("\nPlanned Allocations:")
        for symbol, original_pct in self.original_sizes.items():
            final_pct = min(original_pct * self.position_scale, max_position_pct)
            print(f"{symbol}: {original_pct:.1f}% → {final_pct:.1f}%")
        
        total_allocation = sum(min(pct * self.position_scale, max_position_pct) 
                             for pct in self.original_sizes.values())
        print(f"\nTotal Planned Allocation: {total_allocation:.1f}%")
        
        # Track orders and positions
        self.orders = {}
        self.positions = {}
        self._load_existing_positions()
    
    def _load_existing_positions(self):
        """Load existing positions from Alpaca"""
        try:
            positions = api.list_positions()
            for position in positions:
                self.positions[position.symbol] = {
                    'qty': float(position.qty),
                    'entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc)
                }
            
            if positions:
                print("\nExisting Positions:")
                for symbol, pos in self.positions.items():
                    print(f"{symbol}: {pos['qty']} shares @ ${pos['entry_price']:.2f}")
        
        except Exception as e:
            print(f"Error loading positions: {str(e)}")
    
    def check_signals(self):
        """Check trading plan for entry/exit signals"""
        signals = []
        
        # Check each position in trading plan
        for position in self.trading_plan['trading_plan']['positions']:
            symbol = position['symbol']
            
            try:
                # Get current market data
                data = MarketData.from_json(fetch_market_data(symbol))
                current_price = data.close
                
                # If we don't have a position, check entry conditions
                if symbol not in self.positions:
                    for entry_zone in position['entry']['entry_zones']:
                        price_range = entry_zone['price'].split('-')
                        zone_low = float(price_range[0])
                        zone_high = float(price_range[1])
                        
                        if zone_low <= current_price <= zone_high:
                            # Calculate position size with cap
                            base_size_pct = float(entry_zone['size'].strip('%')) * self.position_scale / 100
                            size_pct = min(base_size_pct, self.max_position_pct / 100)
                            position_value = float(self.account.portfolio_value) * size_pct
                            shares = position_value / current_price
                            
                            signals.append({
                                'type': 'entry',
                                'symbol': symbol,
                                'side': 'buy',
                                'qty': shares,
                                'stop_loss': position['entry']['stop_loss']['price'],
                                'targets': position['entry']['targets']
                            })
                
                # If we have a position, check exit conditions
                elif symbol in self.positions:
                    position_data = self.positions[symbol]
                    
                    # Check stop loss
                    if current_price <= float(position['entry']['stop_loss']['price']):
                        signals.append({
                            'type': 'exit',
                            'symbol': symbol,
                            'side': 'sell',
                            'qty': position_data['qty'],
                            'reason': 'stop_loss'
                        })
                    
                    # Check take profit levels
                    for target in position['entry']['targets']:
                        if current_price >= float(target['price']):
                            # Calculate shares to sell at this target
                            target_pct = float(target['size'].strip('%')) / 100
                            shares_to_sell = position_data['qty'] * target_pct
                            
                            signals.append({
                                'type': 'exit',
                                'symbol': symbol,
                                'side': 'sell',
                                'qty': shares_to_sell,
                                'reason': 'take_profit'
                            })
            
            except Exception as e:
                print(f"Error checking signals for {symbol}: {str(e)}")
        
        return signals
    
    def execute_signals(self, signals: list):
        """Execute trading signals on Alpaca paper account"""
        for signal in signals:
            try:
                if signal['type'] == 'entry':
                    # Calculate take profit prices and quantities
                    take_profits = []
                    remaining_shares = signal['qty']
                    
                    for target in signal['targets']:
                        target_shares = signal['qty'] * (float(target['size'].strip('%')) / 100)
                        take_profits.append({
                            'limit_price': target['price'],
                            'qty': target_shares
                        })
                        remaining_shares -= target_shares
                    
                    # Place bracket order with multiple take profits
                    for i, tp in enumerate(take_profits):
                        # For each target, place a bracket order
                        order = api.submit_order(
                            symbol=signal['symbol'],
                            qty=tp['qty'],
                            side='buy',
                            type='market',
                            time_in_force='day',
                            order_class='bracket',
                            take_profit={
                                'limit_price': tp['limit_price']
                            },
                            stop_loss={
                                'stop_price': signal['stop_loss'],
                                'limit_price': signal['stop_loss'] * 0.99  # 1% below stop for limit
                            }
                        )
                    
                    print(f"\nBracket orders placed for {signal['symbol']}:")
                    for i, tp in enumerate(take_profits):
                        print(f"Target {i+1}: {tp['qty']:.2f} shares @ ${tp['limit_price']:.2f}")
                    print(f"Stop Loss: ${signal['stop_loss']:.2f}")
                    
                elif signal['type'] == 'exit':
                    # Cancel any existing orders for this symbol
                    api.cancel_all_orders_for_symbol(signal['symbol'])
                    
                    # Place market sell order for manual exit
                    order = api.submit_order(
                        symbol=signal['symbol'],
                        qty=signal['qty'],
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    
                    print(f"\nManual exit order placed for {signal['symbol']}:")
                    print(f"Shares: {signal['qty']:.2f}")
                    print(f"Reason: {signal['reason']}")
            
            except Exception as e:
                print(f"Error executing signal for {signal['symbol']}: {str(e)}")
    
    def get_portfolio_status(self):
        """Get current portfolio status"""
        try:
            # Update account info
            self.account = api.get_account()
            
            # Get all positions
            positions = api.list_positions()
            
            status = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': float(self.account.portfolio_value),
                'cash': float(self.account.cash),
                'positions': []
            }
            
            for position in positions:
                status['positions'].append({
                    'symbol': position.symbol,
                    'qty': float(position.qty),
                    'entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc)
                })
            
            return status
        
        except Exception as e:
            print(f"Error getting portfolio status: {str(e)}")
            return None

    def rebalance_portfolio(self):
        """Rebalance portfolio at end of day"""
        try:
            print("\nRebalancing portfolio...")
            
            # Get current positions and prices
            current_positions = {
                pos.symbol: float(pos.market_value)
                for pos in api.list_positions()
            }
            
            total_value = float(self.account.portfolio_value)
            current_allocation = sum(current_positions.values())
            current_allocation_pct = (current_allocation / total_value) * 100
            
            print(f"Current Allocation: {current_allocation_pct:.1f}% (${current_allocation:,.2f})")
            print(f"Target Allocation: {self.target_allocation_pct:.1f}% (${total_value * self.target_allocation_pct/100:,.2f})")
            
            # Always rebalance at end of day to maintain target allocation
            print("Closing positions for next day reallocation...")
            
            # Cancel all existing orders
            api.cancel_all_orders()
            
            # Close all positions
            for symbol in current_positions:
                api.close_position(symbol)
            
            print("Positions closed, ready for fresh entry signals tomorrow")
            
        except Exception as e:
            print(f"Error rebalancing portfolio: {str(e)}")

def place_next_day_orders():
    """Place orders for next day's open"""
    trader = AlpacaTrader('trading_plan.json', target_allocation_pct=90.0, max_position_pct=20.0)
    
    # Cancel any existing orders
    api.cancel_all_orders()
    
    # Close any existing positions
    positions = api.list_positions()
    for position in positions:
        api.close_position(position.symbol)
    
    print("\nPlacing orders for next market open:")
    
    # Check signals and place orders
    signals = trader.check_signals()
    if signals:
        print("\nSetting up bracket orders:")
        trader.execute_signals(signals)
    else:
        print("\nNo entry signals for tomorrow")
    
    # Get pending orders
    orders = api.list_orders(status='open')
    if orders:
        print("\nPending Orders for Tomorrow's Open:")
        for order in orders:
            print(f"{order.symbol}: {order.qty} shares @ Market-on-Open")
            if order.legs:
                print(f"  Take Profit: ${float(order.legs[0].take_profit_price):.2f}")
                print(f"  Stop Loss: ${float(order.legs[0].stop_loss_price):.2f}")

def main():
    """Place orders after market close for next day's open"""
    # Get market hours
    clock = api.get_clock()
    
    if not clock.is_open:
        print("\nMarket closed. Setting up orders for tomorrow's open...")
        place_next_day_orders()
    else:
        print("\nMarket is still open. Wait for market close to place orders.")

if __name__ == "__main__":
    main()
