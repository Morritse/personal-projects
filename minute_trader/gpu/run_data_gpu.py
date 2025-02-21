import pandas as pd
import numpy as np
import json
import time
import cupy as cp
from datetime import datetime, timedelta
from vwap_obv_strategy import VAMEStrategy
from gpu_vwap_obv_strategy import GPUVAMEStrategy
from download_data import SYMBOLS, load_cached_data
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import product

# Load configuration
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

def calculate_score(metrics):
    """Calculate weighted score based on configured metrics"""
    score = 0
    for metric, weight in zip(CONFIG['scoring']['metrics'], CONFIG['scoring']['weights']):
        score += metrics[metric] * weight
    return score

def generate_param_combinations():
    """Generate all parameter combinations from optimization grids"""
    param_names = list(CONFIG['optimization_grids'].keys())
    param_values = list(CONFIG['optimization_grids'].values())
    
    combinations = []
    for values in product(*param_values):
        params = CONFIG['base_params'].copy()
        for name, value in zip(param_names, values):
            params[name] = value
        combinations.append(params)
    
    return combinations

def load_data(symbols):
    """Load data from cache for all symbols with progress bar"""
    data = {}
    print("\nLoading cached data...")
    for symbol in tqdm(symbols, desc="Loading symbols"):
        df = load_cached_data(symbol)
        if df is not None and not df.empty:
            data[symbol] = df
        else:
            print(f"\nNo cached data found for {symbol}")
    return data

def prepare_data(data):
    """Prepare data for strategy testing."""
    print("\nData ranges:")
    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} bars ({df.index[0]} -> {df.index[-1]})")
    
    # Combine data into single DataFrame
    return pd.concat(data.values(), keys=data.keys(), names=['symbol'])

def test_single_symbol_gpu(args):
    """Test strategy on a single symbol using GPU acceleration"""
    symbol, df, test_config = args
    try:
        # Ensure all base parameters are included
        full_config = CONFIG['base_params'].copy()
        full_config.update(test_config)
        strategy = GPUVAMEStrategy(full_config)
        trades = strategy.run(df)
        # Clean up GPU memory after each symbol
        cp.get_default_memory_pool().free_all_blocks()
        return symbol, trades, None
    except Exception as e:
        return symbol, [], f"Error testing {symbol}: {str(e)}"

def test_strategy_gpu(symbol_data, test_config):
    """Test strategy on given data using parallel processing with GPU acceleration"""
    # Prepare data for parallel processing
    symbol_dfs = [(symbol, symbol_data.loc[symbol], test_config) for symbol in symbol_data.index.levels[0]]
    
    # Use all available CPUs
    n_processes = cpu_count()
    
    # Run tests in parallel
    results = {}
    errors = []
    with Pool(n_processes) as pool:
        with tqdm(total=len(symbol_dfs), desc="Testing symbols", leave=False) as pbar:
            for symbol, trades, error in pool.imap_unordered(test_single_symbol_gpu, symbol_dfs):
                if error:
                    errors.append(error)
                results[symbol] = trades
                pbar.update(1)
    
    # Print any errors
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(error)
    
    return results

def calculate_metrics(results):
    """Calculate and return strategy metrics"""
    all_trades = []
    for trades in results.values():
        all_trades.extend([t for t in trades if 'pnl' in t])
    
    if not all_trades:
        return None
        
    pnls = [t['pnl'] for t in all_trades]
    winning_trades = [pnl for pnl in pnls if pnl > 0]
    losing_trades = [pnl for pnl in pnls if pnl <= 0]
    
    # Calculate metrics
    total_pnl = sum(pnls)
    win_rate = len(winning_trades) / len(pnls) if pnls else 0
    profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    # Calculate Sharpe and max drawdown
    returns = pd.Series(pnls)
    sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 1 else 0
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    max_drawdown = np.max(running_max - cumulative)
    
    return {
        'total_pnl': total_pnl,
        'n_trades': len(all_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }

def print_metrics(metrics, title="Strategy Performance"):
    """Print strategy metrics in a formatted way"""
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")
    print(f"Net PnL: ${metrics['total_pnl']:,.2f}")
    print(f"Total Trades: {metrics['n_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Avg Win: ${metrics['avg_win']:,.2f}")
    print(f"Avg Loss: ${metrics['avg_loss']:,.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: ${metrics['max_drawdown']:,.2f}")

def optimize_strategy_gpu(symbol_data):
    """Test different parameter combinations to find optimal settings using GPU acceleration"""
    combinations = generate_param_combinations()
    
    best_metrics = None
    best_params = None
    all_results = []
    
    print("\nOptimizing strategy parameters...")
    print(f"Testing {len(combinations)} combinations")
    
    with tqdm(total=len(combinations), desc="Testing combinations") as pbar:
        for params in combinations:
            # Test strategy with these parameters
            results = test_strategy_gpu(symbol_data, params)
            metrics = calculate_metrics(results)
            
            if metrics is None:
                continue
            
            # Calculate score
            score = calculate_score(metrics)
            
            # Store results
            result = {
                'params': params,
                'metrics': metrics,
                'score': score
            }
            all_results.append(result)
            
            # Update best if better
            if best_metrics is None or score > calculate_score(best_metrics):
                best_metrics = metrics
                best_params = params
            
            pbar.update(1)
            
            # Clean up GPU memory after each combination
            cp.get_default_memory_pool().free_all_blocks()
    
    # Sort and print all results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    print("\nAll combinations (sorted by score):")
    for r in all_results:
        print("\nParameters:")
        for param in CONFIG['optimization_grids'].keys():
            print(f"{param}: {r['params'][param]}")
        print(f"Score: {r['score']:.2f}")
        print(f"PnL: ${r['metrics']['total_pnl']:,.2f}")
        print(f"Trades: {r['metrics']['n_trades']}")
        print(f"Sharpe: {r['metrics']['sharpe']:.2f}")
        print(f"Win Rate: {r['metrics']['win_rate']:.1%}")
    
    print("\nBest parameters found:")
    for param in CONFIG['optimization_grids'].keys():
        print(f"{param}: {best_params[param]}")
    print_metrics(best_metrics, "Best Performance")
    
    return best_params

def compare_performance(symbol_data, params):
    """Compare CPU vs GPU performance"""
    print("\nComparing CPU vs GPU performance...")
    
    # Test CPU version
    cpu_start = time.time()
    cpu_strategy = VAMEStrategy(params)
    cpu_results = {}
    for symbol in symbol_data.index.levels[0]:
        trades = cpu_strategy.run(symbol_data.loc[symbol])
        cpu_results[symbol] = trades
    cpu_time = time.time() - cpu_start
    cpu_metrics = calculate_metrics(cpu_results)
    
    # Test GPU version
    gpu_start = time.time()
    gpu_results = test_strategy_gpu(symbol_data, params)
    gpu_time = time.time() - gpu_start
    gpu_metrics = calculate_metrics(gpu_results)
    
    # Print comparison
    print("\nPerformance Comparison:")
    print(f"{'='*50}")
    print(f"CPU Time: {cpu_time:.2f}s")
    print(f"GPU Time: {gpu_time:.2f}s")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    print("\nCPU Results:")
    print_metrics(cpu_metrics)
    print("\nGPU Results:")
    print_metrics(gpu_metrics)

def main():
    # Load cached data
    print("Loading cached data...")
    data = load_data(SYMBOLS)
    
    if not data:
        print("No cached data found. Please run download_data.py first.")
        return
        
    # Prepare data
    print("Preparing data for analysis...")
    symbol_data = prepare_data(data)
    
    # Optimize strategy
    best_params = optimize_strategy_gpu(symbol_data)
    
    # Compare CPU vs GPU performance
    compare_performance(symbol_data, best_params)
    
    # Save best parameters
    CONFIG['base_params'].update(best_params)
    with open('config.json', 'w') as f:
        json.dump(CONFIG, f, indent=4)
    print("\nBest parameters saved to config.json")

if __name__ == '__main__':
    main()
