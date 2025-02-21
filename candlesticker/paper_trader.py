"""
Paper trading tracker that follows the trading plan and records performance.
"""

import json
from datetime import datetime
import pandas as pd
from market_analyzer import fetch_market_data, MarketData

class PaperTrader:
    def __init__(self, trading_plan_file: str, initial_capital: float = 100000):
        """Initialize paper trader with trading plan and starting capital"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades_history = []
        
        # Load trading plan
        with open(trading_plan_file, 'r') as f:
            self.trading_plan = json.load(f)
    
    def check_entry_signals(self):
        """Check if any positions in the trading plan have triggered entry conditions"""
        signals = []
        
        for position in self.trading_plan['trading_plan']['positions']:
            symbol = position['symbol']
            
            try:
                # Get current market data
                data = MarketData.from_json(fetch_market_data(symbol))
                current_price = data.close
                
                # Check entry zones
                for entry_zone in position['entry']['entry_zones']:
                    price_range = entry_zone['price'].split('-')
                    zone_low = float(price_range[0])
                    zone_high = float(price_range[1])
                    
                    if zone_low <= current_price <= zone_high:
                        signals.append({
                            'symbol': symbol,
                            'type': 'entry',
                            'price': current_price,
                            'size': entry_zone['size'],
                            'stop_loss': position['entry']['stop_loss']['price'],
                            'targets': position['entry']['targets'],
                            'conviction': position['conviction']
                        })
            
            except Exception as e:
                print(f"Error checking entry for {symbol}: {str(e)}")
        
        return signals
    
    def check_exit_signals(self):
        """Check if any current positions have triggered exit conditions"""
        signals = []
        
        for symbol, position in self.positions.items():
            try:
                # Get current market data
                data = MarketData.from_json(fetch_market_data(symbol))
                current_price = data.close
                
                # Check stop loss
                if current_price <= position['stop_loss']:
                    signals.append({
                        'symbol': symbol,
                        'type': 'stop_loss',
                        'price': current_price,
                        'size': '100%'  # Full position exit
                    })
                
                # Check take profit levels
                for target in position['targets']:
                    if current_price >= target['price']:
                        signals.append({
                            'symbol': symbol,
                            'type': 'take_profit',
                            'price': current_price,
                            'size': target['size']
                        })
            
            except Exception as e:
                print(f"Error checking exits for {symbol}: {str(e)}")
        
        return signals
    
    def execute_trade(self, signal: dict):
        """Execute a paper trade based on the signal"""
        timestamp = datetime.now().isoformat()
        
        if signal['type'] == 'entry':
            # Calculate position size
            size_pct = float(signal['size'].strip('%')) / 100
            position_value = self.current_capital * size_pct
            shares = position_value / signal['price']
            
            # Record the trade
            trade = {
                'timestamp': timestamp,
                'symbol': signal['symbol'],
                'type': 'entry',
                'price': signal['price'],
                'shares': shares,
                'value': position_value,
                'stop_loss': signal['stop_loss'],
                'targets': signal['targets'],
                'conviction': signal['conviction']
            }
            
            # Update positions
            self.positions[signal['symbol']] = trade
            
        elif signal['type'] in ['stop_loss', 'take_profit']:
            position = self.positions[signal['symbol']]
            size_pct = float(signal['size'].strip('%')) / 100
            shares_to_sell = position['shares'] * size_pct
            value = shares_to_sell * signal['price']
            
            # Calculate P&L
            entry_price = position['price']
            pnl = (signal['price'] - entry_price) * shares_to_sell
            pnl_pct = (pnl / (entry_price * shares_to_sell)) * 100
            
            # Record the trade
            trade = {
                'timestamp': timestamp,
                'symbol': signal['symbol'],
                'type': signal['type'],
                'price': signal['price'],
                'shares': shares_to_sell,
                'value': value,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            }
            
            # Update positions and capital
            if size_pct == 1:  # Full position exit
                del self.positions[signal['symbol']]
            else:  # Partial exit
                self.positions[signal['symbol']]['shares'] -= shares_to_sell
            
            self.current_capital += pnl
        
        self.trades_history.append(trade)
        return trade
    
    def get_performance_summary(self):
        """Generate performance summary"""
        if not self.trades_history:
            return "No trades executed yet."
        
        df = pd.DataFrame(self.trades_history)
        
        summary = {
            'total_trades': len(df),
            'winning_trades': len(df[df['pnl'] > 0]) if 'pnl' in df.columns else 0,
            'current_capital': self.current_capital,
            'total_return': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'open_positions': len(self.positions),
            'largest_win': df['pnl'].max() if 'pnl' in df.columns else 0,
            'largest_loss': df['pnl'].min() if 'pnl' in df.columns else 0
        }
        
        return summary
    
    def save_trading_record(self):
        """Save trading history and performance to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        record = {
            'generated_at': timestamp,
            'performance_summary': self.get_performance_summary(),
            'open_positions': self.positions,
            'trades_history': self.trades_history
        }
        
        filename = f"paper_trading_record_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(record, f, indent=2)
        
        print(f"\nTrading record saved to {filename}")
        return filename

def main():
    """Example usage of paper trader"""
    # Initialize paper trader with trading plan
    trader = PaperTrader('trading_plan.json')
    
    # Check for entry signals
    entry_signals = trader.check_entry_signals()
    if entry_signals:
        print("\nEntry Signals Found:")
        for signal in entry_signals:
            print(f"\nExecuting entry for {signal['symbol']}:")
            trade = trader.execute_trade(signal)
            print(json.dumps(trade, indent=2))
    
    # Check for exit signals
    exit_signals = trader.check_exit_signals()
    if exit_signals:
        print("\nExit Signals Found:")
        for signal in exit_signals:
            print(f"\nExecuting exit for {signal['symbol']}:")
            trade = trader.execute_trade(signal)
            print(json.dumps(trade, indent=2))
    
    # Save trading record
    trader.save_trading_record()

if __name__ == "__main__":
    main()
