import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from strategy.vwap_obv_strategy import VWAPOBVCrossover
import pytz

class MultiSymbolBacktest:
    def __init__(self, symbols, start_date, end_date, initial_capital=100000):
        """Initialize backtest with multiple symbols."""
        # Initialize API client
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Load strategy config
        with open('final_strategy_params.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize strategy
        self.strategy = VWAPOBVCrossover(self.config)
        
        # Backtest parameters
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Track positions and trades
        self.positions = {}
        self.trades = {symbol: [] for symbol in symbols}
    
    def fetch_data(self, symbol):
        """Fetch historical data for symbol."""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Hour,
            start=self.start_date,
            end=self.end_date,
            adjustment='raw'
        )
        
        try:
            bars = self.data_client.get_stock_bars(request)
            df = bars.df
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol]
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            raise
    
    def run_backtest(self):
        """Run backtest on all symbols."""
        print(f"Starting backtest from {self.start_date} to {self.end_date}")
        print(f"Testing symbols: {', '.join(self.symbols)}")
        print(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Fetch XLV data for regime
        xlv_data = self.fetch_data('XLV')
        
        # Fetch and process each symbol
        symbol_data = {}
        for symbol in self.symbols:
            print(f"\nProcessing {symbol}...")
            symbol_data[symbol] = self.fetch_data(symbol)
        
        # Get common date range
        all_dates = set()
        for df in symbol_data.values():
            all_dates.update(df.index)
        common_dates = sorted(list(all_dates))
        
        # Run simulation
        for date in common_dates:
            # Skip after hours (6:30 AM - 1:00 PM PST)
            if date.hour < 6 or date.hour >= 13:
                continue
            
            # Process each symbol
            for symbol in self.symbols:
                try:
                    # Get data windows
                    symbol_window = symbol_data[symbol][:date].tail(50)
                    xlv_window = xlv_data[:date].tail(50)
                    
                    if len(symbol_window) < 20:  # Need enough data for indicators
                        continue
                    
                    current_price = symbol_window.iloc[-1]['close']
                    
                    # Check exits first
                    if symbol in self.positions:
                        position = self.positions[symbol]
                        params = self.strategy.get_regime_parameters(position['regime'])
                        vwap = self.strategy.calculate_vwap(symbol_window).iloc[-1]
                        mfi = self.strategy.calculate_mfi(symbol_window).iloc[-1]
                        
                        # Check exit conditions
                        hours_held = (date - position['entry_time']).total_seconds() / 3600
                        exit_reason = None
                        
                        if current_price <= position['stop_loss']:
                            exit_reason = 'stop_loss'
                        elif current_price >= position['take_profit']:
                            exit_reason = 'take_profit'
                        elif hours_held >= self.strategy.max_hold_hours:
                            exit_reason = 'max_hold_time'
                        elif current_price > vwap:
                            exit_reason = 'vwap_cross'
                        elif mfi > params['mfi_overbought']:
                            exit_reason = 'mfi_overbought'
                        
                        if exit_reason:
                            # Calculate P&L
                            shares = position['size']
                            entry_price = position['entry_price']
                            pnl = (current_price - entry_price) * shares
                            self.capital += pnl
                            
                            # Record trade
                            trade_record = {
                                'symbol': symbol,
                                'entry_time': position['entry_time'],
                                'entry_price': entry_price,
                                'exit_time': date,
                                'exit_price': current_price,
                                'shares': shares,
                                'pnl': pnl,
                                'return': pnl / (entry_price * shares),
                                'regime': position['regime'],
                                'exit_reason': exit_reason
                            }
                            self.trades[symbol].append(trade_record)
                            
                            # Clear position
                            del self.positions[symbol]
                            continue
                    
                    # Check entries if no position
                    if symbol not in self.positions:
                        # Check entry conditions
                        vwap = self.strategy.calculate_vwap(symbol_window).iloc[-1]
                        mfi = self.strategy.calculate_mfi(symbol_window).iloc[-1]
                        obv = self.strategy.calculate_obv(symbol_window)
                        obv_change = obv.diff().iloc[-1]
                        regime = self.strategy.classify_regime(xlv_window)
                        
                        if (current_price < vwap and 
                            mfi < self.strategy.mfi_entry and 
                            obv_change < 0 and 
                            regime):
                            
                            # Get regime parameters
                            params = self.strategy.get_regime_parameters(regime)
                            
                            # Calculate stop distance
                            atr = self.strategy.calculate_atr(symbol_window).iloc[-1]
                            raw_stop = params['stop_mult'] * atr
                            stop_distance = min(max(raw_stop, self.strategy.min_stop_dollars), 
                                             self.strategy.max_stop_dollars)
                            
                            # Calculate position size
                            risk_amount = self.capital * self.strategy.risk_per_trade
                            position_size = round((risk_amount / stop_distance) * params['position_scale'])
                            
                            # Record position
                            self.positions[symbol] = {
                                'entry_time': date,
                                'entry_price': current_price,
                                'size': position_size,
                                'stop_loss': current_price - stop_distance,
                                'take_profit': current_price + (stop_distance * params['reward_risk']),
                                'regime': regime
                            }
                
                except Exception as e:
                    print(f"Error processing {symbol} at {date}: {e}")
                    continue
        
        # Calculate results
        total_trades = 0
        winning_trades = 0
        total_pnl = 0
        max_drawdown = 0
        peak_capital = self.initial_capital
        
        for symbol in self.symbols:
            symbol_trades = self.trades[symbol]
            if not symbol_trades:
                continue
            
            print(f"\n{symbol} Results:")
            print(f"Total Trades: {len(symbol_trades)}")
            
            symbol_winners = len([t for t in symbol_trades if t['pnl'] > 0])
            symbol_pnl = sum(t['pnl'] for t in symbol_trades)
            
            print(f"Win Rate: {symbol_winners/len(symbol_trades):.1%}")
            print(f"Total P&L: ${symbol_pnl:,.2f}")
            print(f"Average P&L per Trade: ${symbol_pnl/len(symbol_trades):,.2f}")
            
            # Update totals
            total_trades += len(symbol_trades)
            winning_trades += symbol_winners
            total_pnl += symbol_pnl
            
            # Calculate drawdown
            cumulative_pnl = self.initial_capital
            for trade in symbol_trades:
                cumulative_pnl += trade['pnl']
                peak_capital = max(peak_capital, cumulative_pnl)
                drawdown = (peak_capital - cumulative_pnl) / peak_capital
                max_drawdown = max(max_drawdown, drawdown)
        
        print("\nOverall Results:")
        print("-"*50)
        print(f"Total Trades: {total_trades}")
        print(f"Overall Win Rate: {winning_trades/total_trades:.1%}")
        print(f"Total P&L: ${total_pnl:,.2f}")
        print(f"Return on Capital: {total_pnl/self.initial_capital:.1%}")
        print(f"Average P&L per Trade: ${total_pnl/total_trades:,.2f}")
        print(f"Max Drawdown: {max_drawdown:.1%}")
        
        return {
            'trades': self.trades,
            'total_pnl': total_pnl,
            'win_rate': winning_trades/total_trades if total_trades > 0 else 0,
            'max_drawdown': max_drawdown
        }

if __name__ == "__main__":
    # Test healthcare stocks over past year
    symbols = ['JNJ']
    end_date = datetime.now(pytz.timezone('US/Pacific'))
    start_date = end_date - timedelta(days=365)
    
    backtest = MultiSymbolBacktest(symbols, start_date, end_date)
    results = backtest.run_backtest()
