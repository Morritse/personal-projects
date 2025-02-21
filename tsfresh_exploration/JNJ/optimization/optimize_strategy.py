import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from strategy.vwap_obv_strategy_optimizer import VWAPOBVCrossover
from run_strategy import load_data, prepare_data
import multiprocessing as mp
from functools import partial

def calculate_metrics(trades_df, capital=100000):
    """Calculate performance metrics for a set of trades."""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    # Filter sell trades
    sells = trades_df[trades_df['action'] == 'SELL']
    
    # Basic metrics
    total_trades = len(sells)
    winning_trades = len(sells[sells['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_pnl = sells['pnl'].sum()
    avg_pnl = sells['pnl'].mean()
    
    # Create daily PnL series
    daily_pnl = sells.set_index('timestamp')['pnl'].resample('D').sum().fillna(0)
    
    # Calculate Sharpe ratio
    daily_returns = daily_pnl / capital
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

def test_parameters(params, jnj_df, xlv_df):
    """Test a single parameter combination."""
    try:
        strategy = VWAPOBVCrossover(params)
        trades = strategy.run(jnj_df, xlv_df)
        
        if trades:
            trades_df = pd.DataFrame(trades)
            metrics = calculate_metrics(trades_df)
            
            result = {
                **params,
                **metrics
            }
            
            print(f"PnL: ${metrics['total_pnl']:,.2f}, Sharpe: {metrics['sharpe_ratio']:.2f}, Win Rate: {metrics['win_rate']:.2%}")
            
            return result
        return None
    except Exception as e:
        print(f"Error testing parameters: {e}")
        return None

def run_grid_search():
    """Run grid search optimization for strategy parameters."""
    # Load and prepare data
    print("Loading data...")
    jnj_data, xlv_data = load_data()
    jnj_df = prepare_data(jnj_data)
    xlv_df = prepare_data(xlv_data)
    
    # Define focused parameter grid
    param_grid = {
        'MFI Period': [7],                   # Fixed at optimal
        'VWAP Window': [90],                 # Fixed at optimal
        'ATR Period': [3],                   # Fixed
        'Regime Window': [20],               # Fixed
        'Min Stop Dollars': [0.50],          # Fixed
        'Max Stop Dollars': [2.00],          # Fixed
        'Risk Per Trade': [0.01],            # Fixed
        'Max Hold Hours': [24]               # Fixed
    }
    
    # Generate focused regime parameter combinations
    regime_params_grid = []
    
    # Enhanced bear_high_vol parameters
    vwap_distances = [-0.01, -0.015, -0.02]  # Price must be 1-2% below VWAP
    mfi_thresholds = [25, 30, 35]            # MFI oversold levels
    obv_periods = [1, 2, 3]                  # OBV momentum periods
    
    for vwap_dist in vwap_distances:
        for mfi in mfi_thresholds:
            for obv in obv_periods:
                regime_params = {
                    'bear_high_vol': {
                        'position_scale': 1.25,        # Fixed at optimal
                        'reward_risk': 2.3,           # Fixed at optimal
                        'stop_mult': 1.2,             # Fixed
                        'vwap_distance': vwap_dist,   # Variable
                        'mfi_threshold': mfi,         # Variable
                        'obv_period': obv             # Variable
                    },
                    'bull_high_vol': {
                        'position_scale': 0.3,
                        'reward_risk': 1.8,
                        'stop_mult': 1.2
                    },
                    'bear_med_vol': {
                        'position_scale': 0.3,
                        'reward_risk': 2.0,
                        'stop_mult': 1.2
                    },
                    'bull_med_vol': {
                        'position_scale': 0.3,
                        'reward_risk': 1.5,
                        'stop_mult': 1.2
                    },
                    'bull_low_vol': {
                        'position_scale': 0.0,
                        'reward_risk': 1.5,
                        'stop_mult': 1.2
                    },
                    'bear_low_vol': {
                        'position_scale': 0.0,
                        'reward_risk': 1.5,
                        'stop_mult': 1.2
                    }
                }
                regime_params_grid.append(regime_params)
    
    # Generate all parameter combinations
    param_combinations = []
    for values in product(*param_grid.values()):
        base_params = dict(zip(param_grid.keys(), values))
        for regime_params in regime_params_grid:
            params = base_params.copy()
            params['regime_params'] = regime_params
            param_combinations.append(params)
    
    total_combinations = len(param_combinations)
    print(f"Testing {total_combinations} parameter combinations...")
    
    # Use multiprocessing to speed up optimization
    with mp.Pool(processes=mp.cpu_count()) as pool:
        test_func = partial(test_parameters, jnj_df=jnj_df, xlv_df=xlv_df)
        results = list(filter(None, pool.map(test_func, param_combinations)))
    
    if not results:
        print("No valid results found.")
        return None
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by different metrics
    best_pnl = results_df.nlargest(10, 'total_pnl')
    best_sharpe = results_df.nlargest(10, 'sharpe_ratio')
    best_win_rate = results_df.nlargest(10, 'win_rate')
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'optimization_results_{timestamp}.csv', index=False)
    
    # Print summary metrics
    print("\nTop 10 by Total PnL:")
    summary_columns = ['total_pnl', 'sharpe_ratio', 'win_rate', 'regime_params']
    print(best_pnl[summary_columns].to_string())
    
    print("\nTop 10 by Sharpe Ratio:")
    print(best_sharpe[summary_columns].to_string())
    
    print("\nTop 10 by Win Rate:")
    print(best_win_rate[summary_columns].to_string())
    
    # Print best configuration details
    best_config = best_sharpe.iloc[0]
    print("\nBest Configuration (by Sharpe):")
    print(f"MFI Period: {best_config['MFI Period']}")
    print(f"VWAP Window: {best_config['VWAP Window']}")
    print("Regime Parameters:")
    for regime, params in best_config['regime_params'].items():
        print(f"  {regime}:")
        for param, value in params.items():
            print(f"    {param}: {value}")
    
    # Save best parameters
    with open(f'best_params_{timestamp}.txt', 'w') as f:
        f.write("Best Parameters (by Sharpe Ratio):\n\n")
        for key, value in best_config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nFull results saved to: optimization_results_{timestamp}.csv")
    print(f"Best parameters saved to: best_params_{timestamp}.txt")
    
    return results_df

if __name__ == "__main__":
    results = run_grid_search()
