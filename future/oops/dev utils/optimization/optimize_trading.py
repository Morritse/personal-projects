import pandas as pd
import itertools
from typing import Dict, List, Tuple
from fetch_futures_data import FUTURES
from strategy import RefinedFuturesStrategy, analyze_portfolio
from config import MA_STRATEGY, TRADING_STRATEGY_GRID
import numpy as np

def load_data():
    """Load data for all futures contracts"""
    dfs = {}
    for symbol in FUTURES.keys():
        try:
            filename = f"data/{symbol.replace('=', '_')}_daily.csv"
            # Read data with explicit datetime index
            df = pd.read_csv(filename, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            # Ensure required columns exist with proper names
            required_columns = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_columns):
                print(f"Missing required columns in {symbol} data")
                continue
                
            # Ensure data is sorted
            df = df.sort_index()
            
            # Remove any rows with NaN values
            df = df.dropna(subset=required_columns)
            
            if len(df) > 0:
                dfs[symbol] = df
            else:
                print(f"No valid data for {symbol} after cleaning")
                
        except FileNotFoundError:
            print(f"No data file found for {symbol}")
    
    if not dfs:
        raise ValueError("No valid data files found")
        
    print(f"\nLoaded {len(dfs)} instruments:")
    for symbol in sorted(dfs.keys()):
        print(f"  {symbol}: {len(dfs[symbol])} rows from {dfs[symbol].index[0].date()} to {dfs[symbol].index[-1].date()}")
    
    return dfs

def evaluate_strategy(dfs: Dict[str, pd.DataFrame], params: Dict) -> Tuple[Dict, pd.Series]:
    """Evaluate strategy with given parameters"""
    # Combine fixed MA parameters with trading strategy parameters
    strategy_params = {**MA_STRATEGY, **params}
    strategy = RefinedFuturesStrategy(**strategy_params)
    
    # Run strategy on each instrument first
    instrument_results = {}
    for symbol, df in dfs.items():
        df_result, metrics = strategy.run_single_instrument(df)
        instrument_results[symbol] = df_result
    
    # Then analyze portfolio with correlation filtering
    metrics = analyze_portfolio(instrument_results, strategy)
    
    if strategy_params.get('debug', False):
        print("\nIndividual Instrument Results:")
        for symbol, df in instrument_results.items():
            ret = df['strat_returns'].mean() * 252
            vol = df['strat_returns'].std() * np.sqrt(252)
            sharpe = ret / vol if vol > 0 else 0
            print(f"{symbol:6}: Sharpe {sharpe:.2f}, Return {ret:.1%}, Vol {vol:.1%}")
    
    return params, metrics

def generate_parameter_combinations():
    """Generate all possible parameter combinations from the grid"""
    keys = TRADING_STRATEGY_GRID.keys()
    values = TRADING_STRATEGY_GRID.values()
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]

def format_params(params: Dict) -> str:
    """Format parameters for display"""
    parts = []
    if params.get('partial_exit_1', 0) > 0:
        parts.append(f"PE1: {params['partial_exit_1']:.1f}")
    if params.get('partial_exit_2', 0) > 0:
        parts.append(f"PE2: {params['partial_exit_2']:.1f}")
    if params.get('time_stop', 0) > 0:
        parts.append(f"Time Stop: {params['time_stop']}d")
    if 'trailing_stop_factor' in params:
        parts.append(f"Trail: {params['trailing_stop_factor']:.1f}")
    if params.get('adx_threshold'):
        parts.append(f"ADX: {params['adx_threshold']}")
    if 'scaling_mode' in params:
        parts.append(f"Scale: {params['scaling_mode']}")
    if params.get('corr_filter'):
        parts.append(f"Corr: {params['corr_filter']:.2f}")
    return ", ".join(parts)

def main():
    # Load data
    print("Loading futures data...")
    dfs = load_data()
    
    if not dfs:
        print("No data files found. Please run fetch_futures_data.py first.")
        return
    
    # Generate all parameter combinations
    param_combinations = generate_parameter_combinations()
    total_combinations = len(param_combinations)
    print(f"\nRunning grid search over {total_combinations} trading strategy combinations...")
    print("Using fixed MA parameters:")
    for k, v in MA_STRATEGY.items():
        print(f"  {k}: {v}")
    
    # Store results
    results = []
    
    # Run grid search with better organization
    print("\nStarting Grid Search Optimization")
    print("=" * 80)
    print(f"Base MA Strategy Parameters:")
    for k, v in MA_STRATEGY.items():
        print(f"  {k}: {v}")
    print("\nOptimizing Trading Parameters:")
    for k, v in TRADING_STRATEGY_GRID.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    # Run grid search
    for i, params in enumerate(param_combinations, 1):
        print(f"\nCombination {i}/{total_combinations}")
        print("-" * 40)
        print("Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        
        try:
            print("\nRunning backtest...")
            _, metrics = evaluate_strategy(dfs, params)
            
            # Store results with flattened parameters
            result = {
                'annual_return': metrics['annual_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'annual_vol': metrics['annual_vol']
            }
            result.update(params)
            results.append(result)
            
            # Print interim results
            print("\nResults:")
            print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"  Annual Return: {metrics['annual_return']:.2%}")
            print(f"  Annual Vol: {metrics['annual_vol']:.2%}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
            
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
    
    if not results:
        print("\nNo valid results to analyze")
        return
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    # Print top 10 results with more detail
    print("\nTop 10 Strategy Combinations")
    print("=" * 80)
    
    for i in range(min(10, len(results_df))):
        result = results_df.iloc[i]
        print(f"\nRank {i+1}")
        print("-" * 40)
        print("Parameters:")
        for k, v in result.items():
            if k not in ['annual_return', 'sharpe_ratio', 'max_drawdown', 'annual_vol']:
                print(f"  {k}: {v}")
        print("\nPerformance:")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"  Annual Return: {result['annual_return']:.2%}")
        print(f"  Annual Vol: {result['annual_vol']:.2%}")
        print(f"  Max Drawdown: {result['max_drawdown']:.2%}")
    
    # Save results to CSV
    results_df.to_csv('trading_optimization_results.csv', index=False)
    print("\nFull results saved to trading_optimization_results.csv")

if __name__ == "__main__":
    main()
