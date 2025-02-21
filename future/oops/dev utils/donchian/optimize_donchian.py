import pandas as pd
import itertools
from typing import Dict, List, Tuple
from fetch_futures_data import FUTURES
from donchian_strategy import DonchianStrategy, analyze_portfolio

from config import DONCHIAN_GRID

def load_data():
    """Load data for all futures contracts"""
    dfs = {}
    for symbol in FUTURES.keys():
        try:
            filename = f"data/{symbol.replace('=', '_')}_daily.csv"
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            dfs[symbol] = df
        except FileNotFoundError:
            print(f"No data file found for {symbol}")
    return dfs

def evaluate_strategy(dfs: Dict[str, pd.DataFrame], params: Dict) -> Tuple[Dict, pd.Series]:
    """Evaluate strategy with given parameters"""
    strategy = DonchianStrategy(**params)
    metrics = analyze_portfolio(dfs, strategy)
    return params, metrics

def generate_parameter_combinations():
    """Generate all possible parameter combinations from the grid"""
    keys = DONCHIAN_GRID.keys()
    values = DONCHIAN_GRID.values()
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combo)) for combo in combinations]

def format_params(params: Dict) -> str:
    """Format parameters for display"""
    return (f"Channel Period: {params['channel_period']}, "
            f"Vol Window: {params['vol_lookback']}, "
            f"Vol Target: {params['vol_target']:.2f}, "
            f"ATR Stop: {params['stop_atr_multiple']:.1f}")

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
    print(f"\nRunning grid search over {total_combinations} parameter combinations...")
    
    # Store results
    results = []
    
    # Run grid search
    for i, params in enumerate(param_combinations, 1):
        print(f"\nTesting combination {i}/{total_combinations}")
        print(format_params(params))
        
        try:
            params, metrics = evaluate_strategy(dfs, params)
            
            # Store results
            results.append({
                'params': params,
                'annual_return': metrics['annual_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'annual_vol': metrics['annual_vol']
            })
            
            # Print interim results
            print(f"Sharpe: {metrics['sharpe_ratio']:.2f}, "
                  f"Return: {metrics['annual_return']:.2%}, "
                  f"Drawdown: {metrics['max_drawdown']:.2%}")
            
        except Exception as e:
            print(f"Error evaluating parameters: {e}")
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    # Print top 10 results
    print("\nTop 10 Parameter Combinations by Sharpe Ratio:")
    print("=" * 80)
    
    for i in range(min(10, len(results_df))):
        result = results_df.iloc[i]
        print(f"\nRank {i+1}:")
        print(f"Parameters: {format_params(result['params'])}")
        print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"Annual Return: {result['annual_return']:.2%}")
        print(f"Annual Volatility: {result['annual_vol']:.2%}")
        print(f"Maximum Drawdown: {result['max_drawdown']:.2%}")
    
    # Save results to CSV
    results_df.to_csv('donchian_optimization_results.csv', index=False)
    print("\nFull results saved to donchian_optimization_results.csv")

if __name__ == "__main__":
    main()
