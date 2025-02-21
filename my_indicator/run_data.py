import pandas as pd
import numpy as np
from datetime import datetime
from vwap_obv_strategy import VAMEStrategy
from download_data import SYMBOLS, load_cached_data
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import product
from config_bt import config as CONFIG
import math

def calculate_score(metrics):
    """
    Custom objective: we take the product of Sharpe and Total PnL,
    and then use a square root (RMS-like approach).
    You can tweak this formula to suit your needs.
    """
    sharpe = metrics['sharpe']
    pnl    = metrics['total_pnl']
    
    # If either Sharpe or PnL is zero or negative, we can handle it:
    if sharpe <= 0 or pnl <= 0:
        return 0.0  # or some penalty for negative values
    
    # RMS-like approach to Sharpe × PnL:
    # e.g. sqrt( Sharpe × PnL ) to encourage
    # both a good Sharpe ratio and a decent PnL.
    return math.sqrt((sharpe*sharpe) * math.sqrt(pnl))

def expand_param_grid(config):
    """Convert config with arrays into list of individual configs"""
    def _expand_dict(d):
        """Helper function to expand a single level dictionary"""
        param_grid = {}
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively expand nested dictionaries
                expanded = _expand_dict(value)
                param_combinations = []
                keys = expanded.keys()
                values = expanded.values()
                for combo in product(*values):
                    param_combinations.append(dict(zip(keys, combo)))
                param_grid[key] = param_combinations
            elif isinstance(value, (list, np.ndarray)):
                param_grid[key] = value
            else:
                param_grid[key] = [value]
        return param_grid
    
    expanded = _expand_dict(config)
    keys = expanded.keys()
    values = expanded.values()
    
    configs = []
    for combo in product(*values):
        config_instance = {}
        for k, v in zip(keys, combo):
            config_instance[k] = v
        configs.append(config_instance)
    
    return configs

def load_data(symbols):
    """Load data from cache for all symbols with progress bar"""
    data = {}
    print("\nLoading cached data...")
    for symbol in tqdm(symbols, desc="Loading symbols"):
        cache_symbol = symbol.replace('/', '_')
        df = load_cached_data(cache_symbol)
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
    return pd.concat(data.values(), keys=data.keys(), names=['symbol'])

def test_single_symbol(args):
    """Test strategy on a single symbol (for parallel processing)"""
    symbol, df, test_config = args
    try:
        strategy = VAMEStrategy(test_config)
        trades = strategy.run(df)
        return symbol, trades, None
    except Exception as e:
        return symbol, [], f"Error testing {symbol}: {str(e)}"

def test_strategy(symbol_data, test_config):
    """Test strategy on given data using parallel processing"""
    symbol_dfs = [(symbol, symbol_data.loc[symbol], test_config) for symbol in symbol_data.index.levels[0]]
    n_processes = cpu_count()
    results = {}
    errors = []
    
    log_filename = f"trades_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w') as log_file:
        log_file.write("=== Trade Log ===\n\n")
        log_file.write(f"Strategy Parameters:\n")
        for k, v in test_config.items():
            if not isinstance(v, dict):
                log_file.write(f"{k}: {v}\n")
        log_file.write("\n=== Trades ===\n\n")
        
        with Pool(n_processes) as pool:
            with tqdm(total=len(symbol_dfs), desc="Testing symbols", leave=False) as pbar:
                for symbol, trades, error in pool.imap_unordered(test_single_symbol, symbol_dfs):
                    if error:
                        errors.append(error)
                    else:
                        if trades:
                            log_file.write(f"\nSymbol: {symbol}\n")
                            for trade in trades:
                                log_file.write("-" * 50 + "\n")
                                log_file.write(f"Time: {trade['timestamp']}\n")
                                log_file.write(f"Action: {trade['action']}\n")
                                log_file.write(f"Price: ${trade['price']:.2f}\n")
                                log_file.write(f"Size: {trade['size']}\n")
                                if 'pnl' in trade:
                                    log_file.write(f"PnL: ${trade['pnl']:.2f}\n")
                                if 'reason' in trade:
                                    log_file.write(f"Exit Reason: {trade['reason']}\n")
                                if 'regime' in trade:
                                    log_file.write(f"Regime: {trade['regime']}\n")
                                if 'stop_loss' in trade:
                                    log_file.write(f"Stop Loss: ${trade['stop_loss']:.2f}\n")
                                if 'take_profit' in trade:
                                    log_file.write(f"Take Profit: ${trade['take_profit']:.2f}\n")
                    results[symbol] = trades
                    pbar.update(1)
    
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
        return {
            'total_pnl': 0,
            'n_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'sharpe': 0,
            'max_drawdown': 0
        }
        
    pnls = [t['pnl'] for t in all_trades]
    winning_trades = [pnl for pnl in pnls if pnl > 0]
    losing_trades = [pnl for pnl in pnls if pnl <= 0]
    
    total_pnl = sum(pnls)
    win_rate = len(winning_trades) / len(pnls) if pnls else 0
    profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
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

def print_metrics(metrics, params, custom_score=None):
    """Print metrics with parameters, plus your new objective."""
    print("\nParameters:")
    for k, v in params.items():
        if isinstance(v, dict):
            print(f"\n{k}:")
            for sub_k, sub_v in v.items():
                print(f"  {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")
            
    print("\nPerformance:")
    # Original lines:
    print(f"Sharpe: {metrics['sharpe']:.2f}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total PnL: ${metrics['total_pnl']:,.2f}")
    print(f"Total Trades: {metrics['n_trades']}")
    
    # Add your custom score print:
    if custom_score is not None:
        print(f"Custom Score (sqrt(Sharpe * PnL)): {custom_score:.4f}")

def optimize_strategy(symbol_data):
    """Test strategy with parameter combinations"""
    configs = expand_param_grid(CONFIG)
    print(f"\nTesting {len(configs)} parameter combinations...")
    
    best_metrics = None
    best_config  = None
    best_score   = float('-inf')
    
    for config in configs:
        # Run the strategy for these config parameters
        results = test_strategy(symbol_data, config)
        
        # Calculate the usual metrics (PnL, Sharpe, etc.)
        metrics = calculate_metrics(results)
        
        # Use your new custom objective
        score = calculate_score(metrics)
        
        # Optionally, for debug: print out the score for each combo
        # print(f"Config: {config} => Score={score:.4f}, Sharpe={metrics['sharpe']:.2f}, PnL={metrics['total_pnl']:.2f}")
        
        # Keep track of best
        if score > best_score:
            best_score   = score
            best_metrics = metrics
            best_config  = config
    
    # Final printout
    print("\nBest Configuration Found:")
    # Updated call so we can pass in best_score
    print_metrics(best_metrics, best_config, best_score)
    
    return best_config

def main():
    print("Loading cached data...")
    data = load_data(SYMBOLS)
    
    if not data:
        print("No cached data found. Please run download_data.py first.")
        return
        
    print("Preparing data for analysis...")
    symbol_data = prepare_data(data)
    
    best_params = optimize_strategy(symbol_data)
    
    import json
    with open('config.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    print("\nBest parameters saved to config.json")

if __name__ == '__main__':
    main()
