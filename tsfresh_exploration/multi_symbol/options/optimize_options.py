import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from backtest_options import OptionsBacktester
from options_strategy import OptionsStrategy

def test_strategy():
    """Test a single parameter set across multiple symbols."""
    # Test parameters
    params = {
        # Entry Parameters
        'vwap_length': 50,
        'mfi_length': 9,
        'mfi_oversold': 30,
        'mfi_overbought': 70,
        
        # Options Parameters
        'min_dte': 7,
        'max_dte': 21,
        'strike_distance': 0.75,  # Middle ground
        
        # Risk Parameters
        'profit_target': 1.0,     # 100% profit target
        'stop_loss': -0.5,        # 50% stop loss
        
        # Regime Parameters
        'regime_window': 20,
        'vol_percentile': 67
    }
    
    # Test on multiple symbols
    symbols = ['NVDA', 'META', 'AAPL', 'TSLA', 'AMD']
    
    print("\nTesting parameters:")
    print(f"Strike Distance: {params['strike_distance']}")
    print(f"DTE Range: {params['min_dte']}-{params['max_dte']}")
    print(f"Profit Target: {params['profit_target']*100}%")
    print(f"Stop Loss: {params['stop_loss']*100}%")
    print(f"\nTesting on symbols: {', '.join(symbols)}")
    
    try:
        # Initialize backtester
        backtester = OptionsBacktester(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2024, 1, 1),
            account_size=100000
        )
        backtester.strategy = OptionsStrategy(params)
        
        # Run backtest
        trades_df = backtester.run_backtest(symbols)
        
        if trades_df is not None and len(trades_df) > 0:
            # Filter for closed trades
            closed_trades = trades_df[trades_df['action'] == 'SELL']
            
            # Calculate metrics
            returns = closed_trades['return']
            
            # Risk metrics
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            sortino = np.sqrt(252) * returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else np.inf
            max_drawdown = calculate_max_drawdown(closed_trades['pnl'].cumsum())
            
            # Trading metrics
            win_rate = (closed_trades['pnl'] > 0).mean()
            profit_factor = abs(closed_trades[closed_trades['pnl'] > 0]['pnl'].sum() / closed_trades[closed_trades['pnl'] < 0]['pnl'].sum())
            avg_trade = closed_trades['pnl'].mean()
            
            print("\nResults:")
            print("-" * 50)
            print(f"Total Trades: {len(closed_trades)}")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Average Win: ${closed_trades[closed_trades['pnl'] > 0]['pnl'].mean():.2f}")
            print(f"Average Loss: ${closed_trades[closed_trades['pnl'] < 0]['pnl'].mean():.2f}")
            print(f"Total P&L: ${closed_trades['pnl'].sum():.2f}")
            print(f"Average Days Held: {closed_trades['days_held'].mean():.1f}")
            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Max Drawdown: ${max_drawdown:.2f}")
            
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
                        symbol_sharpe = np.sqrt(252) * symbol_trades['return'].mean() / symbol_trades['return'].std()
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
                    regime_sharpe = np.sqrt(252) * regime_trades['return'].mean() / regime_trades['return'].std()
                    print(f"Sharpe: {regime_sharpe:.2f}")
            
            # Save results
            os.makedirs('results', exist_ok=True)
            trades_df.to_csv('results/options_backtest_trades.csv')
            
            return trades_df
    
    except Exception as e:
        print(f"Error in backtest: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve."""
    rolling_max = equity_curve.cummax()
    drawdowns = equity_curve - rolling_max
    return abs(drawdowns.min())

if __name__ == "__main__":
    test_strategy()
