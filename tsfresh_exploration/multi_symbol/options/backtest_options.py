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
        self.strategy = OptionsStrategy({})
        
        # Minimum hold time to avoid day trades
        self.min_hold_days = 1
        
        # Transaction costs
        self.commission_per_contract = 0.65  # TD Ameritrade rate
        self.min_commission = 1.00
        self.slippage_pct = 0.02  # 2% slippage
    
    def load_data(self, symbol):
        """Load historical data from cache."""
        cache_file = f'../data/cache/{symbol.lower()}_data.pkl'  # Fixed path
        if not os.path.exists(cache_file):
            print(f"Looking for data in: {os.path.abspath(cache_file)}")
            raise FileNotFoundError(f"Data not found for {symbol}")
        
        df = pd.read_pickle(cache_file)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        # Filter date range
        df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
        
        if len(df) == 0:
            raise ValueError(f"No data found for {symbol} in specified date range")
            
        return df
    
    def calculate_option_price(self, stock_price, strike, vol, dte, is_entry=True):
        """Calculate theoretical option price using Black-Scholes."""
        # Convert DTE to years
        T = dte / 365
        
        # Risk-free rate (simplified)
        r = 0.04
        
        # Add volatility premium for OTM options (IV skew)
        strike_distance = abs(strike - stock_price) / stock_price
        vol_premium = 1 + (strike_distance * 0.5)  # More realistic skew
        adjusted_vol = vol * vol_premium
        
        # Calculate d1 and d2
        d1 = (np.log(stock_price/strike) + (r + adjusted_vol**2/2)*T) / (adjusted_vol*np.sqrt(T))
        d2 = d1 - adjusted_vol*np.sqrt(T)
        
        # Calculate option price
        call_price = stock_price * norm.cdf(d1) - strike * np.exp(-r*T) * norm.cdf(d2)
        
        # Apply more realistic bid-ask spread based on price and DTE
        if call_price < 1:
            spread = 0.15  # 15% spread for cheap options
        elif call_price < 5:
            spread = 0.10  # 10% spread for mid-price options
        else:
            spread = 0.05  # 5% spread for expensive options
            
        # Slightly wider spreads for shorter DTE
        if dte < 10:
            spread *= 1.2
            
        # Add slippage
        spread += self.slippage_pct
            
        if is_entry:
            price = call_price * (1 + spread)  # Pay more on entry
        else:
            price = call_price * (1 - spread)  # Receive less on exit
            
        # Add commission
        commission = max(self.min_commission, self.commission_per_contract)
        if is_entry:
            price += commission
        else:
            price -= commission
            
        return price
    
    def calculate_option_pnl(self, entry_price, current_price, strike, vol, days_held, initial_dte):
        """Calculate option P&L with realistic pricing."""
        # Calculate remaining DTE
        remaining_dte = initial_dte - days_held
        
        if remaining_dte <= 0:
            # Option expired
            intrinsic = max(0, current_price - strike)
            return intrinsic - entry_price
        
        # Calculate exit price
        exit_price = self.calculate_option_price(
            current_price, 
            strike, 
            vol, 
            remaining_dte,
            is_entry=False
        )
        
        # Apply theta decay (accelerating near expiry)
        base_theta = 0.02  # 2% per day base decay
        if remaining_dte < 7:
            theta_mult = 1.5  # 50% more theta in last week
        else:
            theta_mult = 1.0
            
        theta_decay = base_theta * theta_mult * days_held
        exit_price *= (1 - theta_decay)
        
        return exit_price - entry_price
    
    def calculate_sharpe_ratio(self, trades_df, account_size):
        """Calculate more realistic Sharpe ratio."""
        # Get daily P&L
        trades_df = trades_df.copy()
        trades_df['date'] = pd.to_datetime(trades_df['timestamp']).dt.date
        daily_pnl = trades_df.groupby('date')['pnl'].sum().reindex(
            pd.date_range(self.start_date, self.end_date, freq='D'),
            fill_value=0
        )
        
        # Calculate daily returns on account size
        daily_returns = daily_pnl / account_size
        
        # Annualized Sharpe
        annual_return = daily_returns.mean() * 252
        annual_vol = daily_returns.std() * np.sqrt(252)
        
        if annual_vol == 0:
            return 0
            
        return annual_return / annual_vol
    
    def run_backtest(self, symbols):
        """Run backtest across multiple symbols."""
        all_trades = []
        
        for symbol in symbols:
            print(f"\nBacktesting {symbol}...")
            
            try:
                # Load price data
                df = self.load_data(symbol)
                print(f"Loaded {len(df)} bars of data")
                
                # Calculate indicators
                df['vwap'] = self.strategy.calculate_vwap(df, self.strategy.vwap_length)
                df['mfi'] = self.strategy.calculate_mfi(df, self.strategy.mfi_length)
                df['obv'] = self.strategy.calculate_obv(df)
                df['obv_change'] = df['obv'].diff()
                df['volatility'] = self.strategy.calculate_historical_volatility(df)
                df['rel_vol'] = self.strategy.calculate_relative_volatility(df)
                df['regime'], df['regime_strength'] = self.strategy.determine_regime(df)
                
                # Initialize trades list for this symbol
                trades = []
                
                # Walk through data
                position = None
                entry_idx = None
                for i in range(len(df)):
                    current_bar = df.iloc[i]
                    
                    # Check for exit if in position
                    if position:
                        # Calculate days held
                        days_held = (current_bar.name - position['entry_time']).total_seconds() / (24 * 3600)
                        
                        # Skip if minimum hold time not met
                        if days_held < self.min_hold_days:
                            continue
                        
                        # Calculate current P&L
                        pnl = self.calculate_option_pnl(
                            position['entry_price'],
                            current_bar['close'],
                            position['strike'],
                            current_bar['volatility'],
                            days_held,
                            position['dte']
                        ) * position['size']
                        
                        # Check exit conditions
                        profit_exit = pnl >= position['size'] * self.strategy.profit_target
                        stop_exit = pnl <= position['size'] * self.strategy.stop_loss
                        signal_exit = (
                            current_bar['mfi'] > self.strategy.mfi_overbought or
                            current_bar['close'] > current_bar['vwap'] * 1.01
                        )
                        
                        if profit_exit or stop_exit or signal_exit:
                            # Calculate exit price
                            exit_price = self.calculate_option_price(
                                current_bar['close'],
                                position['strike'],
                                current_bar['volatility'],
                                position['dte'] - days_held,
                                is_entry=False
                            )
                            
                            trades.append({
                                'timestamp': current_bar.name,
                                'symbol': symbol,
                                'action': 'SELL',
                                'price': exit_price,
                                'contract': position['contract'],
                                'strike': position['strike'],
                                'dte': position['dte'],
                                'size': position['size'],
                                'regime': current_bar['regime'],
                                'regime_strength': current_bar['regime_strength'],
                                'pnl': pnl,
                                'return': pnl / position['size'],
                                'exit_reason': 'profit_target' if profit_exit else 'stop_loss' if stop_exit else 'signal',
                                'days_held': days_held
                            })
                            
                            position = None
                            entry_idx = None
                    
                    # Check for entry if no position
                    if not position:
                        # More balanced entry conditions
                        entry_signal = (
                            # Regime requirement - allow both bull and high vol regimes
                            ((current_bar['regime'] == 'bull_high_vol' and current_bar['regime_strength'] > 0.3) or
                             (current_bar['rel_vol'] > 1.1)) and  # Or just high volatility
                            
                            # Price momentum - pullback to VWAP
                            current_bar['close'] < current_bar['vwap'] * 0.997 and  # Moderate pullback
                            
                            # Volume confirmation
                            current_bar['obv_change'] < -100000 and  # Decent volume
                            
                            # Oversold condition
                            current_bar['mfi'] < self.strategy.mfi_oversold
                        )
                        
                        if entry_signal:
                            # Calculate trade parameters
                            current_price = current_bar['close']
                            volatility = current_bar['volatility']
                            rel_vol = current_bar['rel_vol']
                            
                            target_strike = self.strategy.select_strike_price(
                                current_price,
                                volatility,
                                rel_vol
                            )
                            
                            # Calculate position size
                            base_size = self.strategy.calculate_position_size(
                                self.account_size,
                                volatility,
                                current_bar['regime_strength']
                            )
                            
                            # Find available options
                            valid_options = []
                            for dte in range(self.strategy.min_dte, self.strategy.max_dte + 1):
                                strike = target_strike
                                entry_price = self.calculate_option_price(
                                    current_price,
                                    strike,
                                    volatility,
                                    dte,
                                    is_entry=True
                                )
                                
                                valid_options.append({
                                    'strike': strike,
                                    'dte': dte,
                                    'price': entry_price,
                                    'contract': f"{strike:.1f}C_{dte}d"
                                })
                            
                            if valid_options:
                                # Select option with best risk/reward
                                best_option = min(valid_options, key=lambda x: abs(x['strike'] - target_strike))
                                
                                position = {
                                    'entry_time': current_bar.name,
                                    'contract': best_option['contract'],
                                    'entry_price': best_option['price'],
                                    'strike': best_option['strike'],
                                    'dte': best_option['dte'],
                                    'size': base_size
                                }
                                
                                trades.append({
                                    'timestamp': current_bar.name,
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'price': position['entry_price'],
                                    'contract': position['contract'],
                                    'strike': position['strike'],
                                    'dte': position['dte'],
                                    'size': position['size'],
                                    'regime': current_bar['regime'],
                                    'regime_strength': current_bar['regime_strength'],
                                    'underlying_price': current_price,
                                    'target_strike': target_strike,
                                    'volatility': volatility,
                                    'rel_vol': rel_vol
                                })
                                entry_idx = i
                
                print(f"Generated {len(trades)} trades")
                all_trades.extend(trades)
                
            except Exception as e:
                print(f"Error backtesting {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Process results
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            
            # Calculate metrics
            print("\nBacktest Results:")
            print("-" * 50)
            print(f"Total Trades: {len(trades_df)}")
            
            # Filter for closed trades
            closed_trades = trades_df[trades_df['action'] == 'SELL']
            
            if len(closed_trades) > 0:
                win_rate = (closed_trades['pnl'] > 0).mean()
                avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean()
                avg_loss = closed_trades[closed_trades['pnl'] < 0]['pnl'].mean()
                total_pnl = closed_trades['pnl'].sum()
                avg_hold_time = closed_trades['days_held'].mean()
                
                print(f"Win Rate: {win_rate:.1%}")
                print(f"Average Win: ${avg_win:.2f}")
                print(f"Average Loss: ${avg_loss:.2f}")
                print(f"Total P&L: ${total_pnl:.2f}")
                print(f"Average Days Held: {avg_hold_time:.1f}")
                
                # Calculate realistic Sharpe ratio
                sharpe = self.calculate_sharpe_ratio(closed_trades, self.account_size)
                print(f"Sharpe Ratio: {sharpe:.2f}")
                
                # Performance by symbol
                print("\nPerformance by Symbol:")
                for symbol in symbols:
                    symbol_trades = closed_trades[closed_trades['symbol'] == symbol]
                    if len(symbol_trades) > 0:
                        print(f"\n{symbol}:")
                        print(f"Trades: {len(symbol_trades)}")
                        print(f"Win Rate: {(symbol_trades['pnl'] > 0).mean():.1%}")
                        print(f"Average P&L: ${symbol_trades['pnl'].mean():.2f}")
                        if len(symbol_trades) > 1:
                            symbol_sharpe = self.calculate_sharpe_ratio(symbol_trades, self.account_size)
                            print(f"Sharpe: {symbol_sharpe:.2f}")
                
                # Performance by regime
                print("\nPerformance by Regime:")
                for regime in closed_trades['regime'].unique():
                    regime_trades = closed_trades[closed_trades['regime'] == regime]
                    print(f"\n{regime}:")
                    print(f"Trades: {len(regime_trades)}")
                    print(f"Win Rate: {(regime_trades['pnl'] > 0).mean():.1%}")
                    print(f"Average P&L: ${regime_trades['pnl'].mean():.2f}")
                    if len(regime_trades) > 1:
                        regime_sharpe = self.calculate_sharpe_ratio(regime_trades, self.account_size)
                        print(f"Sharpe: {regime_sharpe:.2f}")
            
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
