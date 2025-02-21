import os
import json
import time
import pandas as pd
import numpy as np
from itertools import product
from multiprocessing import Pool, cpu_count
from datetime import datetime
from vwap_obv_strategy_vec import VWAPOBVCrossoverVec

def load_data(symbol: str):
    """Load market data for a symbol."""
    cache_file = f'../data/cache/{symbol.lower()}_data.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Market data not found for {symbol}")
    
    data = pd.read_pickle(cache_file)
    data.index = data.index.tz_localize(None)
    
    # Filter to 2022-2024
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    return data

def calculate_metrics(trades):
    """Calculate performance metrics for a set of trades."""
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    sells = df[df['action'] == 'SELL']
    
    if len(sells) == 0:
        return None
    
    total_pnl = sells['pnl'].sum()
    win_rate = len(sells[sells['pnl'] > 0]) / len(sells)
    
    # Calculate Sharpe ratio
    daily_pnl = sells.set_index('timestamp')['pnl'].resample('D').sum().fillna(0)
    daily_returns = daily_pnl / 100000  # Assuming $100k capital
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate win metrics by regime
    regime_metrics = {}
    for regime in sells['regime'].unique():
        regime_trades = sells[sells['regime'] == regime]
        regime_wins = len(regime_trades[regime_trades['pnl'] > 0])
        regime_metrics[regime] = {
            'trades': len(regime_trades),
            'win_rate': regime_wins / len(regime_trades),
            'total_pnl': regime_trades['pnl'].sum(),
            'avg_pnl': regime_trades['pnl'].mean()
        }
    
    # Calculate exit reason metrics
    exit_metrics = {}
    for reason in sells['exit_reason'].unique():
        reason_trades = sells[sells['exit_reason'] == reason]
        reason_wins = len(reason_trades[reason_trades['pnl'] > 0])
        exit_metrics[reason] = {
            'trades': len(reason_trades),
            'win_rate': reason_wins / len(reason_trades),
            'total_pnl': reason_trades['pnl'].sum(),
            'avg_pnl': reason_trades['pnl'].mean()
        }
    
    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': len(sells),
        'regime_metrics': regime_metrics,
        'exit_metrics': exit_metrics
    }

def evaluate_params(params, symbols):
    """Evaluate a parameter set across multiple symbols."""
    all_trades = []
    symbol_results = []
    
    for symbol in symbols:
        try:
            data = load_data(symbol)
            strategy = VWAPOBVCrossoverVec(params)
            trades = strategy.run(data)
            
            if trades:
                all_trades.extend(trades)  # Collect all trades
                metrics = calculate_metrics(trades)
                if metrics:
                    metrics['symbol'] = symbol
                    symbol_results.append(metrics)
        
        except Exception as e:
            print(f"Error evaluating {symbol}: {e}")
    
    if not symbol_results:
        return None
    
    # Calculate combined metrics across all trades
    combined_metrics = calculate_metrics(all_trades)
    
    # Calculate average metrics across symbols
    df = pd.DataFrame(symbol_results)
    avg_metrics = {
        'avg_pnl': df['total_pnl'].mean(),
        'avg_win_rate': df['win_rate'].mean(),
        'avg_sharpe': df['sharpe'].mean(),
        'avg_drawdown': df['max_drawdown'].mean(),
    }
    
    return {
        'combined_metrics': combined_metrics,  # Metrics for all trades together
        'avg_metrics': avg_metrics,           # Average of individual symbol metrics
        'symbol_results': symbol_results,     # Individual symbol metrics
        'params': params,
        'total_trades': len(all_trades),
        'num_symbols': len(df)
    }

def optimize_all_symbols():
    """Run optimization across all symbols."""
    # Load symbols from config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Parameter grid focused around best Sharpe values
    param_grid = {
        'vwap_length': [40, 50, 60],         # Center around 50
        'mfi_length': [7, 9, 11],            # Center around 9
        'mfi_oversold': [20, 25, 30],        # Test tighter bands
        'mfi_overbought': [70, 75, 80],      # Test tighter bands
        'regime_window': [20],               # Keep constant
        'vol_percentile': [67],              # Keep constant
        'size_factor': [1.0],                # Keep constant
        'max_position': [0.25],              # Keep constant
        'max_hold_hours': [16, 24, 32],      # Test around 24h
        'profit_target': [0.02],             # Keep constant
        'stop_loss': [0.02]                  # Keep constant
    }
    
    # Generate parameter combinations
    param_keys = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_keys]
    combinations = list(product(*param_values))
    
    print(f"Testing {len(combinations)} parameter combinations")
    print(f"Across {len(config['symbols'])} symbols")
    
    # Run optimization
    results = []
    start_time = time.time()
    
    # Use all CPU cores except one
    with Pool(cpu_count() - 1) as pool:
        for i, combo in enumerate(combinations):
            params = dict(zip(param_keys, combo))
            print(f"\nTesting combination {i+1}/{len(combinations)}")
            
            result = evaluate_params(params, config['symbols'])
            if result:
                results.append(result)
                
                # Print progress
                print(f"\nResults for combination {i+1}:")
                print("\nCombined Metrics (All Trades):")
                print(f"Total Trades: {result['total_trades']}")
                print(f"Win Rate: {result['combined_metrics']['win_rate']:.1%}")
                print(f"Sharpe: {result['combined_metrics']['sharpe']:.2f}")
                print(f"Total PnL: ${result['combined_metrics']['total_pnl']:,.2f}")
                
                print("\nAverage Metrics Across Symbols:")
                print(f"Win Rate: {result['avg_metrics']['avg_win_rate']:.1%}")
                print(f"Sharpe: {result['avg_metrics']['avg_sharpe']:.2f}")
                print(f"PnL: ${result['avg_metrics']['avg_pnl']:,.2f}")
                
                # Group symbols by performance
                df = pd.DataFrame(result['symbol_results'])
                df['pnl_per_trade'] = df['total_pnl'] / df['num_trades']
                
                print("\nTop 5 Performing Symbols:")
                top_5 = df.nlargest(5, 'sharpe')
                for _, row in top_5.iterrows():
                    print(f"\n{row['symbol']}:")
                    print(f"  Sharpe: {row['sharpe']:.2f}")
                    print(f"  Win Rate: {row['win_rate']:.1%}")
                    print(f"  PnL/Trade: ${row['pnl_per_trade']:,.2f}")
                    print(f"  Total PnL: ${row['total_pnl']:,.2f}")
                
                print("\nBottom 5 Performing Symbols:")
                bottom_5 = df.nsmallest(5, 'sharpe')
                for _, row in bottom_5.iterrows():
                    print(f"\n{row['symbol']}:")
                    print(f"  Sharpe: {row['sharpe']:.2f}")
                    print(f"  Win Rate: {row['win_rate']:.1%}")
                    print(f"  PnL/Trade: ${row['pnl_per_trade']:,.2f}")
                    print(f"  Total PnL: ${row['total_pnl']:,.2f}")
    
    total_time = time.time() - start_time
    print(f"\nOptimization completed in {total_time:.1f} seconds")
    
    # Save results
    if results:
        # Convert to DataFrame for sorting
        df_results = []
        for r in results:
            row = {
                'params': str(r['params']),
                'total_trades': r['total_trades'],
                'combined_win_rate': r['combined_metrics']['win_rate'],
                'combined_sharpe': r['combined_metrics']['sharpe'],
                'combined_pnl': r['combined_metrics']['total_pnl'],
                'avg_win_rate': r['avg_metrics']['avg_win_rate'],
                'avg_sharpe': r['avg_metrics']['avg_sharpe'],
                'avg_pnl': r['avg_metrics']['avg_pnl']
            }
            df_results.append(row)
        
        df = pd.DataFrame(df_results)
        
        # Sort by Sharpe ratio
        sorted_df = df.sort_values('combined_sharpe', ascending=False)
        
        print("\nTop 10 Parameter Combinations by Sharpe:")
        print(sorted_df.head(10))
        
        # Save detailed results
        os.makedirs('results/optimization', exist_ok=True)
        sorted_df.to_csv('results/optimization/all_symbols_results.csv')
        
        # Save best parameters
        best_params = eval(sorted_df.iloc[0]['params'])
        with open('results/optimization/best_params.json', 'w') as f:
            json.dump(best_params, f, indent=4)
        
        print("\nResults saved to results/optimization/")
    
    return results

if __name__ == "__main__":
    optimize_all_symbols()
