import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from backtest_options import OptionsBacktester

def optimize_parameters():
    """Run parameter optimization focusing on risk-adjusted returns."""
    
    # Parameter grid
    param_grid = {
        # Entry Parameters
        'vwap_length': [20, 50, 100],
        'mfi_length': [7, 9, 14],
        'mfi_oversold': [20, 25, 30],
        'mfi_overbought': [70, 75, 80],
        
        # Options Parameters
        'min_dte': [5, 7, 10],
        'max_dte': [14, 21, 28],
        'strike_distance': [0.5, 0.75, 1.0],  # As multiple of daily volatility
        
        # Position Sizing
        'size_factor': [0.05, 0.093, 0.15],  # Base Kelly fractions
        'max_position': [0.15, 0.25, 0.35],   # Max position size
        
        # Exit Parameters
        'profit_target': [0.5, 1.0, 1.5],     # Multiple of entry price
        'stop_loss': [-0.3, -0.5, -0.7]       # Multiple of entry price
    }


    
    # Test symbols
    symbols = ['NVDA', 'META', 'AAPL', 'TSLA', 'AMD']
    
    # Generate parameter combinations
    param_keys = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_keys]
    combinations = list(product(*param_values))
    
    print(f"Testing {len(combinations)} parameter combinations")
    print(f"Across {len(symbols)} symbols")
    
    # Run optimization in parallel
    results = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for combo in combinations:
            params = dict(zip(param_keys, combo))
            futures.append(
                executor.submit(
                    test_parameters,
                    params,
                    symbols
                )
            )
        
        # Collect results
        for i, future in enumerate(futures):
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"\nCompleted combination {i+1}/{len(combinations)}")
                    print(f"Sharpe: {result['sharpe']:.2f}")
                    print(f"Win Rate: {result['win_rate']:.1%}")
                    print(f"Profit Factor: {result['profit_factor']:.2f}")
            except Exception as e:
                print(f"Error testing combination: {e}")
    
    if results:
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Sort by Sharpe ratio
        df_sorted = df.sort_values('sharpe', ascending=False)
        
        print("\nTop 10 Parameter Combinations:")
        print(df_sorted[['params', 'sharpe', 'sortino', 'win_rate', 'profit_factor', 'max_drawdown']].head(10))
        
        # Save results
        os.makedirs('results/optimization', exist_ok=True)
        df_sorted.to_csv('results/optimization/options_optimization_results.csv')
        
        # Save best parameters
        best_params = df_sorted.iloc[0]['params']
        with open('results/optimization/best_options_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        
        # Generate performance report
        generate_performance_report(df_sorted)
        
        return df_sorted
    
    return None

def test_parameters(params, symbols):
    """Test a single parameter combination."""
    try:
        # Initialize backtester with parameters
        backtester = OptionsBacktester(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2024, 1, 1),
            account_size=100000
        )
        backtester.strategy.params.update(params)
        
        # Run backtest
        trades_df = backtester.run_backtest(symbols)
        
        if trades_df is not None and len(trades_df) > 0:
            # Calculate metrics
            returns = trades_df['return']
            
            # Risk metrics
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            sortino = np.sqrt(252) * returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else np.inf
            max_drawdown = calculate_max_drawdown(trades_df['pnl'].cumsum())
            
            # Trading metrics
            win_rate = (trades_df['pnl'] > 0).mean()
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            avg_trade = trades_df['pnl'].mean()
            
            # Options-specific metrics
            avg_dte = trades_df['dte'].mean()
            avg_hold_time = trades_df['days_held'].mean()
            
            # Calculate regime-specific metrics
            regime_metrics = {}
            for regime in trades_df['regime'].unique():
                regime_trades = trades_df[trades_df['regime'] == regime]
                regime_returns = regime_trades['return']
                regime_metrics[regime] = {
                    'trades': len(regime_trades),
                    'win_rate': (regime_trades['pnl'] > 0).mean(),
                    'sharpe': np.sqrt(252) * regime_returns.mean() / regime_returns.std() if len(regime_trades) > 1 else 0,
                    'avg_pnl': regime_trades['pnl'].mean()
                }
            
            return {
                'params': params,
                'total_trades': len(trades_df),
                'sharpe': sharpe,
                'sortino': sortino,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'avg_trade': avg_trade,
                'avg_dte': avg_dte,
                'avg_hold_time': avg_hold_time,
                'regime_metrics': regime_metrics
            }
    
    except Exception as e:
        print(f"Error testing parameters: {e}")
        return None

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve."""
    rolling_max = equity_curve.cummax()
    drawdowns = equity_curve - rolling_max
    return abs(drawdowns.min())

def generate_performance_report(results_df):
    """Generate detailed performance report."""
    top_10 = results_df.head(10)
    
    # Create report
    report = "Options Strategy Optimization Report\n"
    report += "=" * 50 + "\n\n"
    
    # Overall Statistics
    report += "Overall Statistics:\n"
    report += f"Total Combinations Tested: {len(results_df)}\n"
    report += f"Average Sharpe Ratio: {results_df['sharpe'].mean():.2f}\n"
    report += f"Average Win Rate: {results_df['win_rate'].mean():.1%}\n\n"
    
    # Best Parameters
    best = results_df.iloc[0]
    report += "Best Parameter Combination:\n"
    for param, value in best['params'].items():
        report += f"{param}: {value}\n"
    report += f"\nSharpe Ratio: {best['sharpe']:.2f}\n"
    report += f"Win Rate: {best['win_rate']:.1%}\n"
    report += f"Profit Factor: {best['profit_factor']:.2f}\n"
    report += f"Max Drawdown: ${best['max_drawdown']:,.2f}\n\n"
    
    # Parameter Analysis
    report += "Parameter Analysis:\n"
    for param in best['params'].keys():
        param_performance = results_df.groupby(results_df['params'].apply(lambda x: x[param]))['sharpe'].mean()
        report += f"\n{param}:\n"
        for value, sharpe in param_performance.items():
            report += f"  {value}: {sharpe:.2f}\n"
    
    # Save report
    with open('results/optimization/options_optimization_report.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    optimize_parameters()
