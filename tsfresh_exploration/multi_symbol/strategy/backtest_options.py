import os
import json
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from options_strategy import OptionsStrategy

class OptionsBacktester:
    def __init__(self, start_date=None, end_date=None, account_size=100000):
        self.start_date = start_date or datetime(2022, 1, 1)
        self.end_date = end_date or datetime(2024, 1, 1)
        self.account_size = account_size
        self.strategy = OptionsStrategy(backtest_mode=True)
    
    def load_data(self, symbol):
        """Load historical data from cache."""
        cache_file = f'../data/cache/{symbol.lower()}_data.pkl'
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Data not found for {symbol}")
        
        df = pd.read_pickle(cache_file)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Filter date range
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
        return df
    
    def calculate_option_price(self, stock_price, strike, vol, dte, is_entry=True):
        """Calculate theoretical option price using Black-Scholes."""
        # Convert DTE to years
        T = dte / 365
        
        # Risk-free rate (simplified)
        r = 0.04
        
        # Calculate d1 and d2
        d1 = (np.log(stock_price/strike) + (r + vol**2/2)*T) / (vol*np.sqrt(T))
        d2 = d1 - vol*np.sqrt(T)
        
        # Calculate option price
        call_price = stock_price * norm.cdf(d1) - strike * np.exp(-r*T) * norm.cdf(d2)
        
        # Apply bid-ask spread
        if is_entry:
            return call_price * 1.1  # Pay 10% more on entry
        else:
            return call_price * 0.9  # Receive 10% less on exit
    
    def calculate_option_pnl(self, entry_price, current_price, strike, vol, days_held, initial_dte):
        """Calculate option P&L with realistic pricing."""
        # Calculate remaining DTE
        remaining_dte = initial_dte - days_held
        if remaining_dte <= 0:
            # Option expired
            return max(0, current_price - strike) - entry_price
        
        # Calculate exit price
        exit_price = self.calculate_option_price(
            current_price, 
            strike, 
            vol, 
            remaining_dte,
            is_entry=False
        )
        
        return exit_price - entry_price
    
    def run_backtest(self, symbols):
        """Run backtest across multiple symbols."""
        all_trades = []
        
        for symbol in symbols:
            print(f"\nBacktesting {symbol}...")
            
            try:
                # Load and prepare data
                df = self.load_data(symbol)
                
                # Calculate indicators
                df['vwap'] = self.strategy.base_strategy.calculate_vwap(df, self.strategy.params['vwap_length'])
                df['mfi'] = self.strategy.base_strategy.calculate_mfi(df, self.strategy.params['mfi_length'])
                df['obv'] = self.strategy.base_strategy.calculate_obv(df)
                df['obv_change'] = df['obv'].diff()
                df['hist_vol'] = self.strategy.calculate_historical_volatility(df)
                df['rel_vol'] = self.strategy.calculate_relative_volatility(df)
                df['regime'] = self.strategy.determine_market_regime(df)
                
                # Calculate regime strength
                df['regime_strength'] = 0.0
                bull_mask = df['regime'] == 'bull_high_vol'
                bear_mask = df['regime'] == 'bear_high_vol'
                df.loc[bull_mask, 'regime_strength'] = df.loc[bull_mask, 'rel_vol']
                df.loc[bear_mask, 'regime_strength'] = -df.loc[bear_mask, 'rel_vol']
                
                # Walk through data
                position = None
                for i in range(len(df)):
                    current_bar = df.iloc[i]
                    
                    # Check for exit if in position
                    if position:
                        # Calculate days held (convert timedelta to days)
                        days_held = (current_bar.name - position['entry_date']).total_seconds() / (24 * 3600)
                        
                        # Calculate current P&L
                        pnl = self.calculate_option_pnl(
                            position['entry_price'],
                            current_bar['close'],
                            position['strike'],
                            current_bar['hist_vol'],
                            days_held,
                            position['dte']
                        ) * position['size']
                        
                        # Check exit conditions
                        exit_signal = (
                            days_held >= position['dte'] or  # Expiry
                            current_bar['mfi'] > self.strategy.params['mfi_overbought'] or  # Overbought
                            current_bar['close'] > current_bar['vwap'] * 1.01 or  # Clear VWAP break
                            pnl <= -position['size'] * 0.5 or  # 50% stop loss
                            pnl >= position['size'] * 1.0  # 100% profit target
                        )
                        
                        if exit_signal:
                            trade = {
                                'symbol': symbol,
                                'entry_date': position['entry_date'],
                                'exit_date': current_bar.name,
                                'entry_price': position['entry_price'],
                                'exit_price': current_bar['close'],
                                'strike': position['strike'],
                                'dte': position['dte'],
                                'days_held': days_held,
                                'size': position['size'],
                                'pnl': pnl,
                                'return': pnl / position['size'],
                                'regime': position['regime'],
                                'regime_strength': position['regime_strength']
                            }
                            all_trades.append(trade)
                            position = None
                            continue
                    
                    # Check for entry if no position
                    if not position:
                        # More selective entry conditions
                        entry_signal = (
                            (current_bar['regime'] != 'none' or current_bar['rel_vol'] > 0.6) and  # Stronger regime/vol requirement
                            current_bar['mfi'] < self.strategy.params['mfi_oversold'] and
                            current_bar['close'] < current_bar['vwap'] * 0.99 and  # Clear VWAP break
                            current_bar['obv_change'] < 0 and
                            abs(current_bar['regime_strength']) > 0.4  # Strong regime
                        )
                        
                        if entry_signal:
                            # Calculate trade parameters
                            strike = self.strategy.select_strike(
                                current_bar['close'],
                                current_bar['hist_vol'],
                                current_bar['rel_vol']
                            )
                            dte = self.strategy.get_optimal_expiry(
                                current_bar['hist_vol'],
                                current_bar['rel_vol']
                            )
                            
                            # Adjust position size based on regime strength
                            base_size = self.strategy.calculate_position_size(
                                self.account_size,
                                current_bar['hist_vol'],
                                current_bar['rel_vol']
                            )
                            size = base_size * (1 + abs(current_bar['regime_strength']))
                            
                            # Calculate entry price
                            entry_price = self.calculate_option_price(
                                current_bar['close'],
                                strike,
                                current_bar['hist_vol'],
                                dte,
                                is_entry=True
                            )
                            
                            # Enter position
                            position = {
                                'entry_date': current_bar.name,
                                'entry_price': entry_price,
                                'strike': strike,
                                'dte': dte,
                                'size': size,
                                'regime': current_bar['regime'],
                                'regime_strength': current_bar['regime_strength']
                            }
                
            except Exception as e:
                print(f"Error backtesting {symbol}: {e}")
                continue
        
        # Calculate performance metrics
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            
            print("\nBacktest Results:")
            print("-" * 50)
            print(f"Total Trades: {len(trades_df)}")
            print(f"Win Rate: {(trades_df['pnl'] > 0).mean():.1%}")
            print(f"Average Win: ${trades_df[trades_df['pnl'] > 0]['pnl'].mean():.2f}")
            print(f"Average Loss: ${trades_df[trades_df['pnl'] < 0]['pnl'].mean():.2f}")
            print(f"Total P&L: ${trades_df['pnl'].sum():.2f}")
            print(f"Average Days Held: {trades_df['days_held'].mean():.1f}")
            print(f"Sharpe Ratio: {(trades_df['return'].mean() / trades_df['return'].std()) * np.sqrt(252):.2f}")
            
            # Calculate by regime
            print("\nPerformance by Regime:")
            for regime in trades_df['regime'].unique():
                regime_trades = trades_df[trades_df['regime'] == regime]
                print(f"\n{regime}:")
                print(f"Trades: {len(regime_trades)}")
                print(f"Win Rate: {(regime_trades['pnl'] > 0).mean():.1%}")
                print(f"Average P&L: ${regime_trades['pnl'].mean():.2f}")
                print(f"Sharpe: {(regime_trades['return'].mean() / regime_trades['return'].std()) * np.sqrt(252):.2f}")
            
            # Save results
            os.makedirs('results', exist_ok=True)
            trades_df.to_csv('results/options_backtest_trades.csv')
            
            return trades_df
        
        return None

if __name__ == "__main__":
    # Test symbols
    symbols = ['NVDA', 'META', 'AAPL', 'TSLA', 'AMD']
    
    # Run backtest
    backtester = OptionsBacktester(
        start_date=datetime(2022, 1, 1),
        end_date=datetime(2024, 1, 1),
        account_size=100000
    )
    
    results = backtester.run_backtest(symbols)
