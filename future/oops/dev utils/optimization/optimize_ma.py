import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from fetch_futures_data import FUTURES
from strategy import RefinedFuturesStrategy, analyze_portfolio
from config import MA_GRID

def load_data():
    """Load data for all futures contracts"""
    dfs = {}
    for symbol in FUTURES.keys():
        try:
            filename = f"data/{symbol.replace('=', '_')}_daily.csv"
            # Read CSV and parse dates with UTC=True to handle mixed timezones
            df = pd.read_csv(filename)
            df.index = pd.to_datetime(df['Date'], utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Select only needed columns
            df.sort_index(inplace=True)
            
            # Verify we have enough data
            if len(df) > 200:
                dfs[symbol] = df
            else:
                print(f"Insufficient data for {symbol}: {len(df)} rows")
                
        except FileNotFoundError:
            print(f"No data file found for {symbol}")
        except Exception as e:
            print(f"Error loading {symbol}: {e}")
    
    if not dfs:
        raise ValueError("No valid data files found")
    
    print(f"\nLoaded {len(dfs)} instruments:")
    for symbol in sorted(dfs.keys()):
        print(f"  {symbol}: {len(dfs[symbol])} rows ({dfs[symbol].index[0].date()} to {dfs[symbol].index[-1].date()})")
    
    return dfs

def evaluate_strategy(dfs: Dict[str, pd.DataFrame], params: Dict) -> Tuple[Dict, pd.Series]:
    """Evaluate strategy with given parameters"""
    strategy = RefinedFuturesStrategy(**params)
    metrics = analyze_portfolio(dfs, strategy)
    return params, metrics

def generate_parameter_combinations():
    """Generate all possible parameter combinations from the grid"""
    # Base parameters that stay constant
    base_params = {
        'debug': False,
        'partial_exit_1': 0.75,
        'partial_exit_2': 1.0,
        'time_stop': 10,
        'trailing_stop_factor': 2.0,
        'adx_threshold': None,
        'scaling_mode': 'none',
        'corr_filter': False
    }
    
    # Generate combinations from MA_GRID
    param_combinations = []
    for fast in MA_GRID['lookback_fast']:
        for slow in MA_GRID['lookback_slow']:
            if fast < slow:  # Only test when fast MA is shorter than slow MA
                for vol_lookback in MA_GRID['vol_lookback']:
                    for vol_target in MA_GRID['vol_target']:
                        for stop_atr in MA_GRID['stop_atr_multiple']:
                            params = base_params.copy()
                            params['lookback_fast'] = fast
                            params['lookback_slow'] = slow
                            params['vol_lookback'] = vol_lookback
                            params['vol_target'] = vol_target
                            params['stop_atr_multiple'] = stop_atr
                            param_combinations.append(params)
    
    return param_combinations

def format_params(params: Dict) -> str:
    """Format parameters for display"""
    return (f"Fast MA: {params['lookback_fast']}, "
            f"Slow MA: {params['lookback_slow']}, "
            f"Vol Window: {params['vol_lookback']}, "
            f"Vol Target: {params['vol_target']:.3f}, "
            f"ATR Stop: {params['stop_atr_multiple']:.1f}")

def main():
    # Load data
    print("Loading futures data...")
    try:
        dfs = load_data()
    except ValueError as e:
        print(f"Error: {e}")
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
            
            # Validate metrics
            if pd.isna(metrics['sharpe_ratio']) or pd.isna(metrics['annual_return']):
                print("Warning: Invalid metrics - skipping combination")
                continue
                
            # Store results
            results.append({
                'lookback_fast': params['lookback_fast'],
                'lookback_slow': params['lookback_slow'],
                'vol_lookback': params['vol_lookback'],
                'vol_target': params['vol_target'],
                'stop_atr_multiple': params['stop_atr_multiple'],
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
            continue
    
    if not results:
        print("\nNo valid results to analyze")
        return
    
    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Sort by Sharpe ratio
    results_df = results_df.sort_values('sharpe_ratio', ascending=False)
    
    # Print top 10 results with cleaner formatting
    print("\nTop 10 Parameter Combinations by Sharpe Ratio:")
    print("=" * 80)
    
    for i in range(min(10, len(results_df))):
        result = results_df.iloc[i]
        print(f"\nRank {i+1}:")
        print("Parameters:")
        print(f"  Fast MA: {result['lookback_fast']}")
        print(f"  Slow MA: {result['lookback_slow']}")
        print(f"  Vol Window: {result['vol_lookback']}")
        print(f"  Vol Target: {result['vol_target']:.3f}")
        print(f"  ATR Stop: {result['stop_atr_multiple']:.1f}")
        print("\nPerformance:")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"  Annual Return: {result['annual_return']:.2%}")
        print(f"  Annual Volatility: {result['annual_vol']:.2%}")
        print(f"  Maximum Drawdown: {result['max_drawdown']:.2%}")
    
    # Save results to CSV
    results_df.to_csv('ma_optimization_results.csv', index=False)
    print("\nFull results saved to ma_optimization_results.csv")

if __name__ == "__main__":
    main()
