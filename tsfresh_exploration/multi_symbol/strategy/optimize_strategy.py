import os
import json
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from download_data import ensure_data_downloaded
from vwap_obv_strategy import VWAPOBVCrossover

def load_data(symbol: str):
    """Load market data for a symbol."""
    cache_file = f'../data/cache/{symbol.lower()}_data.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Market data not found for {symbol}")
    
    data = pd.read_pickle(cache_file)
    
    # Convert UTC timestamps to naive timestamps
    data.index = data.index.tz_localize(None)
    
    # Filter to 2022-2024
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    return data

def calculate_metrics(trades_df, capital=100000):
    """Calculate performance metrics for a set of trades."""
    if trades_df.empty:
        return None
    
    # Filter sell trades
    sells = trades_df[trades_df['action'] == 'SELL']
    
    # Basic metrics
    total_trades = len(sells)
    winning_trades = len(sells[sells['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_pnl = sells['pnl'].sum()
    avg_pnl = sells['pnl'].mean()
    
    # Calculate daily returns
    daily_pnl = sells.set_index('timestamp')['pnl'].resample('D').sum().fillna(0)
    daily_returns = daily_pnl / capital
    
    # Calculate Sharpe ratio
    daily_std = daily_returns.std()
    daily_mean = daily_returns.mean()
    sharpe_ratio = np.sqrt(252) * (daily_mean / daily_std) if daily_std > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def generate_param_combinations(param_ranges):
    """Generate all parameter combinations from ranges."""
    param_values = {}
    for section, params in param_ranges.items():
        for param, ranges in params.items():
            values = np.arange(ranges['min'], ranges['max'] + ranges['step'], ranges['step'])
            param_values[f"{section}_{param}"] = values
    
    # Generate all combinations
    keys = list(param_values.keys())
    combinations = list(product(*[param_values[k] for k in keys]))
    
    return [dict(zip(keys, combo)) for combo in combinations]

def evaluate_params(params, symbols, results_dir='results/optimization'):
    """Evaluate a parameter set across multiple symbols."""
    all_metrics = []
    
    for symbol in symbols:
        try:
            # Load data
            data = load_data(symbol)
            
            # Create strategy instance with params
            strategy_params = {
                'vwap_length': params['vwap_length'],
                'mfi_length': params['mfi_length'],
                'mfi_oversold': params['mfi_oversold'],
                'mfi_overbought': params['mfi_overbought'],
                'regime_window': params['regime_window'],
                'vol_percentile': params['regime_vol_percentile'],
                'size_factor': params['position_size_factor'],
                'max_position': params['position_max_position'],
                'max_hold_hours': params['exit_max_hold_hours'],
                'profit_target': params['exit_profit_target'],
                'stop_loss': params['exit_stop_loss']
            }
            
            strategy = VWAPOBVCrossover(strategy_params)
            
            # Run strategy
            trades = strategy.run(data)
            
            if trades:
                trades_df = pd.DataFrame(trades)
                metrics = calculate_metrics(trades_df)
                if metrics:
                    metrics['symbol'] = symbol
                    all_metrics.append(metrics)
        
        except Exception as e:
            print(f"Error evaluating {symbol}: {e}")
            continue
    
    if not all_metrics:
        return None
    
    # Combine metrics across symbols
    df = pd.DataFrame(all_metrics)
    
    # Calculate aggregate scores
    avg_metrics = {
        'avg_win_rate': df['win_rate'].mean(),
        'avg_sharpe': df['sharpe_ratio'].mean(),
        'avg_drawdown': df['max_drawdown'].mean(),
        'total_pnl': df['total_pnl'].sum(),
        'num_symbols': len(df),
        'params': params
    }
    
    return avg_metrics

def optimize_strategy(config_path='config.json', num_iterations=1000):
    """Run optimization across all symbols."""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create results directory
    results_dir = 'results/optimization'
    os.makedirs(results_dir, exist_ok=True)
    
    # Ensure we have data for all symbols
    ensure_data_downloaded(
        config['symbols'],
        config['alpaca']['api_key'],
        config['alpaca']['api_secret'],
        base_path='..'
    )
    
    # Generate parameter combinations
    param_combinations = generate_param_combinations(config['strategy_params'])
    print(f"Generated {len(param_combinations)} parameter combinations")
    
    # Run optimization
    results = []
    for i, params in enumerate(param_combinations):
        print(f"\nTesting combination {i+1}/{len(param_combinations)}")
        metrics = evaluate_params(params, config['symbols'])
        if metrics:
            results.append(metrics)
        
        # Save intermediate results
        if (i + 1) % 10 == 0:
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(results_dir, 'optimization_results.csv'))
            
            # Save best params so far
            best_idx = df['avg_sharpe'].idxmax()
            best_params = df.iloc[best_idx]['params']
            with open(os.path.join(results_dir, 'best_params.json'), 'w') as f:
                json.dump(best_params, f, indent=4)
    
    # Final results
    df = pd.DataFrame(results)
    
    # Sort by different metrics
    metrics = ['avg_sharpe', 'avg_win_rate', 'total_pnl']
    for metric in metrics:
        sorted_df = df.sort_values(metric, ascending=False)
        sorted_df.to_csv(os.path.join(results_dir, f'results_by_{metric}.csv'))
        
        # Save top params
        top_params = sorted_df.iloc[0]['params']
        with open(os.path.join(results_dir, f'best_params_by_{metric}.json'), 'w') as f:
            json.dump(top_params, f, indent=4)
    
    print("\nOptimization complete!")
    print(f"Results saved in {results_dir}")
    
    return df

if __name__ == "__main__":
    optimize_strategy()
