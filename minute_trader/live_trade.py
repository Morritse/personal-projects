import os
import time
import pytz
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca_trade_api import REST, TimeFrame
from dotenv import load_dotenv
from vectorized_strategy import precompute_all_indicators, fill_regime_column
from portfolio_strategy import run_portfolio_strategy
from config import config as CONFIG
from download_data import SYMBOLS
from tabulate import tabulate
from colorama import Fore, Style, init

# Load Alpaca credentials
load_dotenv()
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_SECRET_KEY')
api = REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')

# Constants will be loaded from config
TOTAL_CAPITAL = None
MAX_POSITIONS = None
MAX_CAPITAL_USAGE = 0.9  # Maximum % of total capital to use

EST = pytz.timezone('US/Eastern')
PST = pytz.timezone('US/Pacific')
MARKET_OPEN = datetime.strptime('09:30', '%H:%M').time()
MARKET_CLOSE = datetime.strptime('16:00', '%H:%M').time()

def setup_logging():
    """Setup logging to file and console"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Setup logging
    log_file = f'logs/live_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

class LiveTrader:
    def __init__(self):
        # Initialize colorama
        init()
        
        self.api = api
        # Use first value from any parameter lists in config
        self.config = {}
        for key, value in CONFIG.items():
            if isinstance(value, list):
                self.config[key] = value[0]  # Take first value
            else:
                self.config[key] = value
        
        # Set constants from config
        global TOTAL_CAPITAL, MAX_POSITIONS
        TOTAL_CAPITAL = self.config.get('Initial Capital', 100000)
        MAX_POSITIONS = self.config.get('Max Positions', 5)
        
        self.positions = {}  # Track open positions
        self.available_capital = TOTAL_CAPITAL
        self.logger = setup_logging()
        
        # Minimal startup log
        self.logger.info("Starting...")
        
    def get_current_positions(self):
        """Update positions from Alpaca"""
        positions = {}
        alpaca_positions = self.api.list_positions()
        for position in alpaca_positions:
            positions[position.symbol] = {
                'size': int(position.qty),
                'entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'market_value': float(position.market_value)
            }
        self.positions = positions
        self.available_capital = TOTAL_CAPITAL - sum(p['market_value'] for p in positions.values())
        return positions
    
    def get_historical_data(self, symbol, lookback_bars=400):
        """Get historical minute bars for analysis"""
        end = datetime.now(EST)
        start = end - timedelta(days=3)  # Get 3 days of minute data
        
        try:
            bars = self.api.get_bars(
                symbol,
                TimeFrame.Minute,
                start.strftime('%Y-%m-%d'),
                end.strftime('%Y-%m-%d')
            ).df
            
            if isinstance(bars.index, pd.MultiIndex):
                bars = bars.xs(symbol, level=0)
            
            return bars
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def analyze_symbol(self, symbol):
        """Run strategy analysis on a symbol"""
        # Get historical data
        df = self.get_historical_data(symbol)
        if df is None or len(df) < 200:  # Need enough bars for analysis
            return None
            
        # Prepare data
        df_pre = precompute_all_indicators(
            df,
            regime_window=self.config.get('Regime Window', 20),
            volatility_percentile=self.config.get('Volatility Percentile', 67)
        )
        
        # Fill regime
        df_reg = fill_regime_column(df_pre)
        
        return df_reg
    
    def check_entry_signals(self):
        """Check all symbols for entry signals"""
        signals = []
        table_data = []
        
        # Calculate capital allocation
        remaining_positions = MAX_POSITIONS - len(self.positions)
        if remaining_positions <= 0:
            self.logger.info("Maximum positions reached, no new entries allowed")
            return []
        
        # Calculate available capital (respect MAX_CAPITAL_USAGE limit)
        max_usable_capital = TOTAL_CAPITAL * MAX_CAPITAL_USAGE
        true_available = min(self.available_capital, max_usable_capital)
        
        # Calculate per-trade capital based on max allocation
        max_allocation = self.config.get('Max Allocation', 0.2)  # 20% default
        capital_per_trade = true_available * max_allocation
        
        for symbol in SYMBOLS:
            # Skip if we already have a position
            if symbol in self.positions:
                continue
                
            # Analyze symbol
            df = self.analyze_symbol(symbol)
            if df is None:
                continue
            
            # Get latest bar and check conditions
            current_bar = df.iloc[-1]
            price_below_vwap = current_bar["close"] < current_bar["vwap"]
            obv_falling = current_bar["obv_diff"] < 0
            mfi_oversold = current_bar["mfi"] < self.config.get("mfi_entry", 30)
            
            # Add row to table data
            table_data.append([
                symbol,
                f"${current_bar['close']:.2f} vs ${current_bar['vwap']:.2f}",
                Fore.GREEN + "✓" + Style.RESET_ALL if price_below_vwap else Fore.RED + "✗" + Style.RESET_ALL,
                f"{current_bar['obv_diff']:.0f}",
                Fore.GREEN + "✓" + Style.RESET_ALL if obv_falling else Fore.RED + "✗" + Style.RESET_ALL,
                f"{current_bar['mfi']:.1f}",
                Fore.GREEN + "✓" + Style.RESET_ALL if mfi_oversold else Fore.RED + "✗" + Style.RESET_ALL
            ])
            
            # Run portfolio strategy on recent data
            df_recent = df.tail(50)  # Use last 50 bars for regime calculation
            df_recent = precompute_all_indicators(
                df_recent,
                regime_window=self.config.get('Regime Window', 20),
                volatility_percentile=self.config.get('Volatility Percentile', 67)
            )
            df_recent = fill_regime_column(df_recent)
            symbol_data = {symbol: df_recent}
            trades = run_portfolio_strategy(symbol_data, self.config)
            
            # Check if we got any trades
            if trades:
                latest_trade = trades[-1]
                if latest_trade['action'] == 'BUY':
                    # Skip if missing required fields
                    if not all(k in latest_trade for k in ['price', 'size', 'stop_loss', 'take_profit']):
                        self.logger.warning(f"Skipping {symbol} trade - missing required fields")
                        continue
                        
                    signals.append({
                        'symbol': symbol,
                        'action': 'buy',
                        'size': latest_trade['size'],
                        'price': latest_trade['price'],
                        'stop_loss': latest_trade['stop_loss'],
                        'take_profit': latest_trade['take_profit'],
                        'regime': latest_trade.get('regime')
                    })
                    
                    self.logger.info(f"TRADE: BUY {symbol} @ ${latest_trade['price']:.2f}")
        
        # Print table after collecting all data
        if table_data:
            table = tabulate(table_data, headers=["Symbol", "Price vs VWAP", "✓/✗", "OBV_diff", "✓/✗", "MFI", "✓/✗"], tablefmt="grid")
            self.logger.info("\n" + table)
        
        return signals
    
    def check_exit_signals(self):
        """Check existing positions for exit signals"""
        signals = []
        
        if self.positions:
            self.logger.info("Checking exit signals for current positions")
        
        for symbol, position in self.positions.items():
            # Get latest data
            df = self.analyze_symbol(symbol)
            if df is None:
                continue
            
            # Run portfolio strategy on recent data
            df_recent = df.tail(50)  # Use last 50 bars for regime calculation
            df_recent = precompute_all_indicators(
                df_recent,
                regime_window=self.config.get('Regime Window', 20),
                volatility_percentile=self.config.get('Volatility Percentile', 67)
            )
            df_recent = fill_regime_column(df_recent)
            symbol_data = {symbol: df_recent}
            trades = run_portfolio_strategy(symbol_data, self.config)
            
            # Check if we got any trades
            if trades:
                latest_trade = trades[-1]
                if latest_trade['action'] == 'SELL':
                    signals.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'size': latest_trade['size'],
                        'reason': latest_trade.get('reason', 'technical')
                    })
        
        return signals
    
    def execute_trades(self, signals):
        """Execute trade signals"""
        for signal in signals:
            try:
                if signal['action'] == 'buy':
                    # Place buy order
                    self.api.submit_order(
                        symbol=signal['symbol'],
                        qty=signal['size'],
                        side='buy',
                        type='limit',
                        time_in_force='day',
                        limit_price=signal['price'],
                        order_class='bracket',
                        stop_loss={'stop_price': signal['stop_loss']},
                        take_profit={'limit_price': signal['take_profit']}
                    )
                    self.logger.info(f"BUY {signal['symbol']}: {signal['size']} shares @ ${signal['price']:.2f}")
                    self.logger.info(f"Stop: ${signal['stop_loss']:.2f}, Target: ${signal['take_profit']:.2f}")
                elif signal['action'] == 'sell':
                    # Place sell order
                    self.api.close_position(signal['symbol'])
                    self.logger.info(f"SELL {signal['symbol']}: {signal['size']} shares - {signal['reason']}")
            except Exception as e:
                self.logger.error(f"Error executing {signal['action']} for {signal['symbol']}: {e}")
    
    def run_trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting live trading...")
        
        while True:
            # Get current time in both timezones
            now_pst = datetime.now(PST)
            now_est = now_pst.astimezone(EST)
            current_time = now_est.time()
            
            if current_time < MARKET_OPEN or current_time > MARKET_CLOSE:
                next_run = now_est
                if current_time > MARKET_CLOSE:
                    next_run += timedelta(days=1)
                next_run = next_run.replace(
                    hour=9, minute=30, second=0, microsecond=0
                )
                
                # Convert next run time back to PST for local display
                next_run_pst = next_run.astimezone(PST)
                
                sleep_seconds = (next_run - now_est).total_seconds()
                self.logger.info(f"Market closed. Sleeping until {next_run_pst} PST ({next_run} EST)")
                time.sleep(sleep_seconds)
                continue
            
            try:
                # Update positions silently
                self.get_current_positions()
                
                # Check for entry and exit signals
                entry_signals = self.check_entry_signals()
                exit_signals = self.check_exit_signals()
                
                # Execute any trades
                self.execute_trades(exit_signals + entry_signals)
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
            
            # Sleep until next check (30 seconds)
            next_check = now_est + timedelta(seconds=5)
            next_check_pst = next_check.astimezone(PST)
            sleep_seconds = (next_check - now_est).total_seconds()
            time.sleep(sleep_seconds)

if __name__ == '__main__':
    trader = LiveTrader()
    trader.run_trading_loop()
