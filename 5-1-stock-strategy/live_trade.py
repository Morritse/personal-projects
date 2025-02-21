import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
import time
import os
from typing import Dict, List, Optional
from unified import UnifiedStrategy
from config import config as CONFIG
from bear_config import bear_config as BEAR_CONFIG
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Load Alpaca credentials
load_dotenv()
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')

class LiveTrader:
    def __init__(self, symbols: List[str]):
        """Initialize live trader with list of symbols"""
        self.symbols = symbols
        
        # Initialize both strategies with non-array config values
        bull_config = {k: v[0] if isinstance(v, list) else v for k, v in CONFIG.items()}
        
        # For bear strategy, double the initial capital to account for margin
        bear_config = {k: v[0] if isinstance(v, list) else v for k, v in BEAR_CONFIG.items()}
        bear_config["Initial Capital"] = bear_config.get("Initial Capital", 100000) * 2
        
        # Handle nested regime_params
        if 'regime_params' in bull_config:
            bull_config['regime_params'] = {
                regime: {k: v[0] if isinstance(v, list) else v 
                        for k, v in params.items()}
                for regime, params in bull_config['regime_params'].items()
            }
        if 'regime_params' in bear_config:
            bear_config['regime_params'] = {
                regime: {k: v[0] if isinstance(v, list) else v 
                        for k, v in params.items()}
                for regime, params in bear_config['regime_params'].items()
            }
        
        self.bull_strategy = UnifiedStrategy(bull_config)
        self.bear_strategy = UnifiedStrategy(bear_config)
        
        # Internal state
        self.last_update_time = None
        self.est_tz = pytz.timezone('US/Eastern')
        
        # Data management
        self.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        
        # Keep enough 1-min bars for 5-min resampling and indicator calculation
        # For regime calculation we need:
        # - 20-25 bars for regime window (5-min bars)
        # - Extra bars for volatility percentile calculation
        # So for 100 5-min bars we need 500 1-min bars
        self.lookback_minutes = 500
        
        # Initialize DataFrames for each symbol
        self.one_min_bars: Dict[str, pd.DataFrame] = {}
        self.five_min_bars: Dict[str, pd.DataFrame] = {}
        self.processed_bars: Dict[str, pd.DataFrame] = {}  # Bars with indicators
        
        for symbol in symbols:
            self.one_min_bars[symbol] = pd.DataFrame(columns=self.columns)
            self.one_min_bars[symbol].set_index("timestamp", inplace=True)
            
            self.five_min_bars[symbol] = pd.DataFrame(columns=self.columns)
            self.five_min_bars[symbol].set_index("timestamp", inplace=True)
            
            self.processed_bars[symbol] = {
                'bull': pd.DataFrame(columns=self.columns).set_index("timestamp"),
                'bear': pd.DataFrame(columns=self.columns).set_index("timestamp")
            }
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now(self.est_tz)
        current_time = now.time()
        
        # Market hours 9:30 AM - 4:00 PM ET
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0).time()
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0).time()
        
        # Check if it's a weekday and within market hours
        return (now.weekday() < 5 and  # Monday = 0, Friday = 4
                market_open <= current_time < market_close)
    
    def update_bars(self, new_data: Dict[str, Dict]):
        """
        Update bars with new market data and compute indicators
        Args:
            new_data: Dict of symbol -> Dict of OHLCV data for 1-minute bars
        """
        current_time = datetime.now(self.est_tz)
        
        for symbol, data in new_data.items():
            # Update 1-minute bars
            df = pd.DataFrame([data])
            df.set_index("timestamp", inplace=True)
            self.one_min_bars[symbol] = pd.concat([self.one_min_bars[symbol], df])
            self.one_min_bars[symbol] = self.one_min_bars[symbol].iloc[-self.lookback_minutes:]
            
            # Resample to 5-minute bars
            df_5min = self.one_min_bars[symbol].resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            self.five_min_bars[symbol] = df_5min
            
            # Compute indicators on 5-minute bars
            if not df_5min.empty:
                # Process for bull strategy
                df_processed = self.bull_strategy.precompute_all_indicators(
                    df_5min,
                    regime_window=self.bull_strategy.config.get("Regime Window", 20),
                    volatility_percentile=self.bull_strategy.config.get("Volatility Percentile", 67),
                    vwap_length=self.bull_strategy.config.get("VWAP Window", 50),
                    mfi_length=self.bull_strategy.config.get("MFI Period", 9)
                )
                df_processed = self.bull_strategy.fill_regime_column(df_processed)
                
                # Process for bear strategy
                df_processed_bear = self.bear_strategy.precompute_all_indicators(
                    df_5min,
                    regime_window=self.bear_strategy.config.get("Regime Window", 20),
                    volatility_percentile=self.bear_strategy.config.get("Volatility Percentile", 67),
                    vwap_length=self.bear_strategy.config.get("VWAP Window", 50),
                    mfi_length=self.bear_strategy.config.get("MFI Period", 9)
                )
                df_processed_bear = self.bear_strategy.fill_regime_column(df_processed_bear)
                
                # Store both bull and bear processed bars
                self.processed_bars[symbol] = {
                    'bull': df_processed,
                    'bear': df_processed_bear
                }
        
        self.last_update_time = current_time
    
    def process_current_bars(self):
        """Process current bars through both strategies"""
        if not self.is_market_open():
            print("Market is closed")
            return
        
        # Get latest 1-minute bar data with indicators from 5-minute bars
        current_data = {}
        for symbol in self.symbols:
            if (not self.one_min_bars[symbol].empty and 
                symbol in self.processed_bars and 
                not self.processed_bars[symbol]['bull'].empty):
                # Get latest 1-minute bar
                latest_1min_bull = self.one_min_bars[symbol].iloc[-1].copy()
                latest_1min_bear = self.one_min_bars[symbol].iloc[-1].copy()
                
                # Get indicators from latest 5-minute bar for both strategies
                latest_5min_bull = self.processed_bars[symbol]['bull'].iloc[-1]
                latest_5min_bear = self.processed_bars[symbol]['bear'].iloc[-1]
                
                # Add indicator values to 1-minute bars
                for col in latest_5min_bull.index:
                    if col not in self.columns:  # Only add indicator columns
                        latest_1min_bull[col] = latest_5min_bull[col]
                        latest_1min_bear[col] = latest_5min_bear[col]
                
                # Use appropriate indicators for each strategy
                current_data[symbol] = latest_1min_bull  # For bull strategy
                current_data[f"{symbol}_bear"] = latest_1min_bear  # For bear strategy
        
        # Process through both strategies
        if current_data:
            # Process bull strategy normally
            bull_data = {k: v for k, v in current_data.items() if not k.endswith('_bear')}
            self.bull_strategy.process_bar(self.last_update_time, bull_data)
            
            # For bear strategy, adjust capital based on margin
            bear_data = {k.replace('_bear', ''): v for k, v in current_data.items() if k.endswith('_bear')}
            # Save current capital
            orig_capital = self.bear_strategy.capital
            # Double available capital for margin
            self.bear_strategy.capital *= 2
            # Process bar
            self.bear_strategy.process_bar(self.last_update_time, bear_data)
            # Restore actual capital
            self.bear_strategy.capital = orig_capital
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions from both strategies"""
        positions = {}
        positions.update({f"BULL_{k}": v for k, v in self.bull_strategy.positions.items()})
        positions.update({f"BEAR_{k}": v for k, v in self.bear_strategy.positions.items()})
        return positions
    
    def get_portfolio_value(self) -> Dict[str, float]:
        """Get current portfolio value for both strategies"""
        # Get latest prices from 1-minute bars
        current_prices = {}
        for symbol, df in self.one_min_bars.items():
            if not df.empty:
                current_prices[symbol] = df.iloc[-1]["close"]
        
        return {
            "bull": self.bull_strategy.update_portfolio_value(current_prices),
            "bear": self.bear_strategy.update_portfolio_value(current_prices)
        }
    
    def print_status(self):
        """Print current trading status"""
        print("\n=== Trading Status ===")
        print(f"Time: {self.last_update_time}")
        
        portfolio_values = self.get_portfolio_value()
        print("\nPortfolio Values:")
        print(f"Bull Strategy: ${portfolio_values['bull']:,.2f}")
        print(f"Bear Strategy: ${portfolio_values['bear']:,.2f}")
        print(f"Total: ${(portfolio_values['bull'] + portfolio_values['bear']):,.2f}")
        
        # Print detailed status for each symbol
        print("\nSymbol Status:")
        for symbol in self.symbols:
            if symbol in self.processed_bars:
                latest_bull = self.processed_bars[symbol]['bull'].iloc[-1]
                latest_bear = self.processed_bars[symbol]['bear'].iloc[-1]
                
                # Use bear indicators for regime since they're more sensitive to downtrends
                latest = latest_bear
                print(f"\n{symbol}:")
                print(f"Last Update: {latest.name}")
                print(f"MFI: {latest['mfi']:.1f}")
                print(f"VWAP: ${latest['vwap']:.2f}")
                print(f"Close: ${latest['close']:.2f}")
                print(f"OBV Diff: {latest['obv_diff']:.0f}")
                
                # Detailed regime info
                print("\nRegime Details:")
                print(f"Smoothed Returns (Ann.): {latest['smoothed_returns']*100:.1f}%")
                print(f"Volatility (Ann.): {latest['volatility']*100:.1f}%")
                print(f"Vol Threshold: {latest['vol_threshold']*100:.1f}%")
                is_high_vol = latest['volatility'] > latest['vol_threshold']
                print(f"High Vol: {is_high_vol}")
                print(f"Final Regime: {latest['regime'] if not pd.isna(latest['regime']) else 'None'}")
                
                # Check entry conditions for both strategies
                if latest['regime'] == 'bull_high_vol':
                    bull_mfi_cond = latest['mfi'] < self.bull_strategy.mfi_entry
                    bull_price_cond = latest['close'] > latest['vwap']
                    bull_obv_cond = latest['obv_diff'] > 0
                    print("Bull Conditions:")
                    print(f"  MFI < {self.bull_strategy.mfi_entry}: {bull_mfi_cond}")
                    print(f"  Price > VWAP: {bull_price_cond}")
                    print(f"  OBV Rising: {bull_obv_cond}")
                
                if latest['regime'] == 'bear_high_vol':
                    bear_mfi_cond = latest['mfi'] > self.bear_strategy.entry_mfi
                    bear_price_cond = latest['close'] < latest['vwap']
                    bear_obv_cond = latest['obv_diff'] < 0
                    print("Bear Conditions:")
                    print(f"  MFI > {self.bear_strategy.entry_mfi}: {bear_mfi_cond}")
                    print(f"  Price < VWAP: {bear_price_cond}")
                    print(f"  OBV Falling: {bear_obv_cond}")
        
        positions = self.get_current_positions()
        if positions:
            print("\nCurrent Positions:")
            for pos_id, pos in positions.items():
                strategy, symbol = pos_id.split("_", 1)
                print(f"\n{strategy} {symbol}:")
                print(f"  {'Short' if pos['is_short'] else 'Long'} {pos['size']} shares")
                print(f"  Entry: ${pos['entry_price']:.2f}")
                print(f"  Stop: ${pos['stop_loss']:.2f}")
                print(f"  Target: ${pos['take_profit']:.2f}")
                
                # Calculate current P&L using latest 1-minute price
                if not self.one_min_bars[symbol].empty:
                    current_price = self.one_min_bars[symbol].iloc[-1]["close"]
                    if pos['is_short']:
                        pnl = (pos['entry_price'] - current_price) * pos['size']
                    else:
                        pnl = (current_price - pos['entry_price']) * pos['size']
                    print(f"  Current P&L: ${pnl:.2f}")
                    
                    # Show current indicators
                    if symbol in self.processed_bars:
                        # Use bear indicators for positions since we care about downside risk
                        latest_indicators = self.processed_bars[symbol]['bear'].iloc[-1]
                        print(f"  Current MFI: {latest_indicators['mfi']:.1f}")
                        print(f"  Regime: {latest_indicators['regime']}")
        else:
            print("\nNo open positions")
    
    def get_market_data(self) -> Dict[str, Dict]:
        """Fetch latest market data from Alpaca"""
        try:
            # Get current minute's bars
            now = datetime.now(self.est_tz)
            start_min = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
            end_min = now.replace(second=0, microsecond=0)
            
            # Convert to UTC for Alpaca API
            start_str = start_min.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = end_min.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            new_data = {}
            for symbol in self.symbols:
                try:
                    bars = api.get_bars(
                        symbol,
                        tradeapi.TimeFrame.Minute,
                        start=start_str,
                        end=end_str
                    ).df
                    
                    if not bars.empty:
                        # Get the latest bar
                        latest = bars.iloc[-1]
                        new_data[symbol] = {
                            "timestamp": bars.index[-1].tz_convert(self.est_tz),
                            "open": float(latest['open']),
                            "high": float(latest['high']),
                            "low": float(latest['low']),
                            "close": float(latest['close']),
                            "volume": float(latest['volume'])
                        }
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {str(e)}")
                    continue
            
            return new_data
        
        except Exception as e:
            print(f"Error in get_market_data: {str(e)}")
            return {}

    def initialize_data(self):
        """Load initial historical data"""
        try:
            # Get enough historical 1-minute bars for indicator calculation
            end_time = datetime.now(self.est_tz)
            start_time = end_time - timedelta(minutes=self.lookback_minutes)
            
            # Convert to UTC for Alpaca API
            start_str = start_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
            end_str = end_time.astimezone(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            print("\nLoading initial historical data...")
            for symbol in self.symbols:
                try:
                    bars = api.get_bars(
                        symbol,
                        tradeapi.TimeFrame.Minute,
                        start=start_str,
                        end=end_str
                    ).df
                    
                    if not bars.empty:
                        # Convert bars to our format
                        df = pd.DataFrame({
                            'open': bars['open'],
                            'high': bars['high'],
                            'low': bars['low'],
                            'close': bars['close'],
                            'volume': bars['volume']
                        }, index=bars.index.tz_convert(self.est_tz))
                        
                        self.one_min_bars[symbol] = df
                        print(f"Loaded {len(df)} bars for {symbol}")
                except Exception as e:
                    print(f"Error loading historical data for {symbol}: {str(e)}")
                    continue
            
            print("\nInitializing indicators...")
            # Initialize indicators
            for symbol in self.symbols:
                if not self.one_min_bars[symbol].empty:
                    print(f"Processing {symbol}...")
                    try:
                        self.update_bars({
                            symbol: {
                                "timestamp": self.one_min_bars[symbol].index[-1],
                                "open": float(self.one_min_bars[symbol]['open'].iloc[-1]),
                                "high": float(self.one_min_bars[symbol]['high'].iloc[-1]),
                                "low": float(self.one_min_bars[symbol]['low'].iloc[-1]),
                                "close": float(self.one_min_bars[symbol]['close'].iloc[-1]),
                                "volume": float(self.one_min_bars[symbol]['volume'].iloc[-1])
                            }
                        })
                        print(f"Processed {symbol} successfully")
                    except Exception as e:
                        print(f"Error processing {symbol}: {str(e)}")
                        print(f"Data for {symbol}:")
                        print(self.one_min_bars[symbol].tail())
        
        except Exception as e:
            print(f"Error in initialize_data: {str(e)}")

    def run(self):
        """Main trading loop"""
        print("\nStarting live trading with both Bull and Bear strategies")
        print(f"Trading symbols: {', '.join(self.symbols)}")
        print("\nBull Strategy Parameters:")
        for k, v in self.bull_strategy.config.items():
            print(f"{k}: {v}")
        print("\nBear Strategy Parameters:")
        for k, v in self.bear_strategy.config.items():
            print(f"{k}: {v}")
        
        try:
            print("\nInitializing historical data...")
            # Load initial historical data
            self.initialize_data()
            print("\nHistorical data initialized successfully")
            
            print("\nChecking market status...")
            is_open = self.is_market_open()
            print(f"Market is {'open' if is_open else 'closed'}")
        
            while True:
                if self.is_market_open():
                    print("\nFetching market data...")
                    # Get latest market data
                    new_data = self.get_market_data()
                    if new_data:
                        print(f"Got data for {len(new_data)} symbols")
                        # Update bars with new data
                        print("Updating bars...")
                        self.update_bars(new_data)
                        print("Bars updated")
                        
                        # Process updated bars
                        print("Processing bars...")
                        self.process_current_bars()
                        print("Bars processed")
                        
                        # Print status after each update
                        print("\nPrinting status...")
                        self.print_status()
                    
                    # Sleep until next minute
                    now = datetime.now(self.est_tz)
                    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
                    sleep_seconds = (next_minute - now).total_seconds()
                    time.sleep(max(1, sleep_seconds))  # At least 1 second to avoid hammering
                else:
                    print("Market is closed. Waiting for market open...")
                    time.sleep(60)  # Check every minute
        
        except KeyboardInterrupt:
            print("\nStopping live trading...")
            # Clean up any resources, close positions if needed
        
        except Exception as e:
            print(f"\nError in live trading: {str(e)}")
            raise

def main():
    # Use same symbols as download_data.py
    from download_data import SYMBOLS
    
    trader = LiveTrader(SYMBOLS)
    trader.run()

if __name__ == "__main__":
    main()
