import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class Position:
    symbol: str
    strike: float
    expiry: datetime
    entry_price: float
    size: int
    entry_time: datetime
    order_id: str
    oco_id: str
    regime: str
    
class LiveOptionsTrader:
    def __init__(self, api_key: str, api_secret: str, paper_trading: bool = True):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize TradeStation API client
        self.client = self.setup_client(api_key, api_secret, paper_trading)
        
        # Strategy parameters (will load from optimization results)
        self.params = self.load_strategy_params()
        
        # Active positions
        self.positions: Dict[str, Position] = {}
        
        # Market data streams
        self.data_streams = {}
        
        # Thread pool for monitoring
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.monitoring = True
    
    def setup_logging(self):
        """Configure logging with detailed formatting."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/options_trader.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_client(self, api_key: str, api_secret: str, paper_trading: bool):
        """Initialize TradeStation API client with proper configuration."""
        # TODO: Implement TradeStation API client setup
        # This will be implemented once we have API credentials
        pass
    
    def load_strategy_params(self) -> dict:
        """Load optimized strategy parameters."""
        try:
            with open('results/optimization/best_options_params.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning("No optimized parameters found, using defaults")
            return {
                'vwap_length': 50,
                'mfi_length': 9,
                'mfi_oversold': 30,
                'mfi_overbought': 70,
                'regime_window': 20,
                'vol_percentile': 67
            }
    
    def place_entry_order(self, symbol: str, strike: float, expiry: datetime, 
                         size: int, regime: str) -> Optional[Position]:
        """Place option entry order with OCO safety net."""
        try:
            # Calculate option details
            entry_price = self.get_option_price(symbol, strike, expiry)
            
            # Place main entry order
            entry_order = {
                "Symbol": symbol,
                "Strike": strike,
                "Expiry": expiry.strftime('%Y-%m-%d'),
                "Action": "BUY_TO_OPEN",
                "Quantity": size,
                "Type": "MARKET"
            }
            
            entry_response = self.client.place_order(entry_order)
            entry_id = entry_response['OrderID']
            
            # Place OCO safety net
            oco_order = {
                "Type": "OCO",
                "Orders": [
                    {
                        "Symbol": symbol,
                        "Strike": strike,
                        "Expiry": expiry.strftime('%Y-%m-%d'),
                        "Action": "SELL_TO_CLOSE",
                        "Quantity": size,
                        "Type": "LIMIT",
                        "Price": entry_price * 2.0  # 100% profit target
                    },
                    {
                        "Symbol": symbol,
                        "Strike": strike,
                        "Expiry": expiry.strftime('%Y-%m-%d'),
                        "Action": "SELL_TO_CLOSE",
                        "Quantity": size,
                        "Type": "STOP",
                        "Price": entry_price * 0.5  # 50% stop loss
                    }
                ]
            }
            
            oco_response = self.client.place_order(oco_order)
            oco_id = oco_response['OrderID']
            
            # Create position object
            position = Position(
                symbol=symbol,
                strike=strike,
                expiry=expiry,
                entry_price=entry_price,
                size=size,
                entry_time=datetime.now(),
                order_id=entry_id,
                oco_id=oco_id,
                regime=regime
            )
            
            # Start monitoring this position
            self.positions[symbol] = position
            self.executor.submit(self.monitor_position, position)
            
            self.logger.info(f"Entered position: {symbol} {strike} {expiry}")
            return position
            
        except Exception as e:
            self.logger.error(f"Error placing entry order: {e}")
            return None
    
    def monitor_position(self, position: Position):
        """Actively monitor position for exit conditions."""
        symbol = position.symbol
        
        while self.monitoring and symbol in self.positions:
            try:
                # Get current market data
                current_data = self.get_market_data(symbol)
                
                # Check technical conditions
                if self.check_exit_conditions(current_data, position):
                    self.exit_position(position, "Technical exit")
                    break
                
                # Check DTE
                days_to_expiry = (position.expiry - datetime.now()).days
                if days_to_expiry <= 1:
                    self.exit_position(position, "Near expiry")
                    break
                
                # Sleep briefly
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error monitoring position {symbol}: {e}")
                # Don't break - keep monitoring despite errors
    
    def check_exit_conditions(self, data: dict, position: Position) -> bool:
        """Check if technical exit conditions are met."""
        return (
            data['mfi'] > self.params['mfi_overbought'] or
            data['close'] > data['vwap'] * 1.01 or
            self.calculate_pnl(position) >= position.entry_price * 1.0 or  # 100% profit
            self.calculate_pnl(position) <= position.entry_price * -0.5    # 50% loss
        )
    
    def exit_position(self, position: Position, reason: str):
        """Exit position and cancel OCO orders."""
        try:
            # Place exit order
            exit_order = {
                "Symbol": position.symbol,
                "Strike": position.strike,
                "Expiry": position.expiry.strftime('%Y-%m-%d'),
                "Action": "SELL_TO_CLOSE",
                "Quantity": position.size,
                "Type": "MARKET"
            }
            
            self.client.place_order(exit_order)
            
            # Cancel OCO safety net
            self.client.cancel_order(position.oco_id)
            
            # Calculate final P&L
            final_pnl = self.calculate_pnl(position)
            
            self.logger.info(
                f"Exited position: {position.symbol} {position.strike} "
                f"Reason: {reason} P&L: ${final_pnl:,.2f}"
            )
            
            # Remove from active positions
            del self.positions[position.symbol]
            
        except Exception as e:
            self.logger.error(f"Error exiting position: {e}")
    
    def calculate_pnl(self, position: Position) -> float:
        """Calculate current P&L for position."""
        try:
            current_price = self.get_option_price(
                position.symbol, 
                position.strike, 
                position.expiry
            )
            return (current_price - position.entry_price) * position.size
        except Exception as e:
            self.logger.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def get_option_price(self, symbol: str, strike: float, expiry: datetime) -> float:
        """Get current option price from TradeStation."""
        # TODO: Implement actual API call
        pass
    
    def get_market_data(self, symbol: str) -> dict:
        """Get current market data including technicals."""
        # TODO: Implement actual API call
        pass
    
    def shutdown(self):
        """Gracefully shutdown the trader."""
        self.logger.info("Shutting down trader...")
        self.monitoring = False
        
        # Exit all positions
        for position in list(self.positions.values()):
            self.exit_position(position, "Shutdown")
        
        # Cleanup
        self.executor.shutdown(wait=True)
        self.logger.info("Trader shutdown complete")

if __name__ == "__main__":
    # Load credentials
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize trader
    trader = LiveOptionsTrader(
        api_key=config['tradestation']['api_key'],
        api_secret=config['tradestation']['api_secret'],
        paper_trading=True
    )
    
    try:
        # Main trading loop would go here
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        trader.shutdown()
