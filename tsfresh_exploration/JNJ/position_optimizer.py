import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from strategy.vwap_obv_strategy import VWAPOBVCrossover
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
            'max_drawdown': 0,
            'regime_contribution': {}
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
    
    # Calculate returns relative to capital
    daily_returns = daily_pnl / capital
    cumulative_returns = (1 + daily_returns).cumprod()
    total_return = (cumulative_returns.iloc[-1] - 1) if len(cumulative_returns) > 0 else 0
    
    # Calculate Sharpe ratio
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 else 0
    
    # Calculate max drawdown
    rolling_max = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdowns.min() if len(drawdowns) > 0 else 0
    
    # Calculate regime contribution
    regime_stats = {}
    for regime in sells['regime'].unique():
        regime_trades = sells[sells['regime'] == regime]
        regime_pnl = regime_trades['pnl'].sum()
        regime_stats[regime] = {
            'trades': len(regime_trades),
            'win_rate': len(regime_trades[regime_trades['pnl'] > 0]) / len(regime_trades),
            'avg_pnl': regime_trades['pnl'].mean(),
            'total_pnl': regime_pnl,
            'contribution': regime_pnl / total_pnl if total_pnl != 0 else 0
        }
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'regime_contribution': regime_stats
    }

def test_parameters(params, jnj_df, xlv_df):
    """Test a single parameter combination."""
    try:
        strategy = VWAPOBVCrossover(params)
        trades = strategy.run(jnj_df, xlv_df)
        
        if trades:
            trades_df = pd.DataFrame(trades)
            metrics = calculate_metrics(trades_df)
            
            # Calculate regime balance score
            regime_stats = metrics['regime_contribution']
            bear_contrib = sum(stats['contribution'] for regime, stats in regime_stats.items() if 'bear' in regime)
            bull_contrib = sum(stats['contribution'] for regime, stats in regime_stats.items() if 'bull' in regime)
            
            # Target 60/40 bear/bull split
            balance_score = 1 - abs(0.6 - bear_contrib) - abs(0.4 - bull_contrib)
            
            result = {
                **params,
                'total_pnl': metrics['total_pnl'],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'win_rate': metrics['win_rate'],
                'max_drawdown': metrics['max_drawdown'],
                'total_trades': metrics['total_trades'],
                'avg_trade_pnl': metrics['avg_pnl'],
                'balance_score': balance_score,
                'regime_contribution': metrics['regime_contribution']
            }
            
            print(f"\nParameters:")
            print(f"Bear Scale: {params['regime_params']['bear_high_vol']['position_scale']:.2f}x")
            print(f"Bull Scale: {params['regime_params']['bull_high_vol']['position_scale']:.2f}x")
            print(f"Bear R/R: {params['regime_params']['bear_high_vol']['reward_risk']:.1f}")
            print(f"Bull R/R: {params['regime_params']['bull_high_vol']['reward_risk']:.1f}")
            print(f"Stop Mult: {params['regime_params']['bear_high_vol']['stop_mult']:.1f}")
            print(f"Return: {metrics['total_return']:.1%}")
            print(f"PnL: ${metrics['total_pnl']:,.2f}")
            print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"Win Rate: {metrics['win_rate']:.1%}")
            print(f"Balance Score: {balance_score:.2f}")
            print("---")
            
            return result
        return None
    except Exception as e:
        print(f"Error testing parameters: {e}")
        return None

def run_position_optimization():
    """Run optimization focused on position sizing and reward/risk."""
    # Load and prepare data
    print("Loading data...")
    jnj_data, xlv_data = load_data()
    jnj_df = prepare_data(jnj_data)
    xlv_df = prepare_data(xlv_data)
    
    # Parameter grid for position sizing and reward/risk
    param_grid = {
        'bear_scale': [1.5, 1.75, 2.0, 2.25],           # Test higher bear sizing
        'bull_scale': [0.75, 1.0, 1.25, 1.5],           # Test higher bull sizing
        'bear_reward_risk': [2.0, 2.5, 3.0, 3.5],       # Test higher bear targets
        'bull_reward_risk': [1.5, 2.0, 2.5, 3.0],       # Test higher bull targets
        'stop_mult': [1.0, 1.2, 1.4, 1.6]               # Test wider trailing stops
    }
    
    # Fixed parameters (using all optimized values)
    fixed_params = {
        'VWAP Window': 50,                    # Optimized
        'MFI Period': 9,                      # Optimized
        'Regime Window': 20,                  # Optimized
        'ATR Period': 2,                      # Optimized
        'Min Stop Dollars': 1.00,             # Optimized
        'Max Stop Dollars': 2.50,             # Optimized
        'Risk Per Trade': 0.025,              # Optimized
        'Max Hold Hours': 36,                 # Optimized
        'mfi_entry': 30,                      # Optimized
        'bear_exit': 55,                      # Optimized
        'bull_exit': 75                       # Optimized
    }
    
    # Generate parameter combinations
    param_combinations = []
    for bear_scale, bull_scale, bear_rr, bull_rr, stop_mult in product(
        param_grid['bear_scale'],
        param_grid['bull_scale'],
        param_grid['bear_reward_risk'],
        param_grid['bull_reward_risk'],
        param_grid['stop_mult']
    ):
        params = {
            **fixed_params,
            'regime_params': {
                'bear_high_vol': {
                    'position_scale': bear_scale,
                    'reward_risk': bear_rr,
                    'stop_mult': stop_mult,
                    'mfi_overbought': 55,     # Optimized
                    'trailing_stop': True
                },
                'bull_high_vol': {
                    'position_scale': bull_scale,
                    'reward_risk': bull_rr,
                    'stop_mult': stop_mult,
                    'mfi_overbought': 75,     # Optimized
                    'trailing_stop': True
                },
                'bear_med_vol': {
                    'position_scale': 0.0,
                    'reward_risk': 2.0,
                    'stop_mult': stop_mult
                },
                'bull_med_vol': {
                    'position_scale': 0.0,
                    'reward_risk': 1.5,
                    'stop_mult': stop_mult
                },
                'bull_low_vol': {
                    'position_scale': 0.0,
                    'reward_risk': 1.5,
                    'stop_mult': stop_mult
                },
                'bear_low_vol': {
                    'position_scale': 0.0,
                    'reward_risk': 1.5,
                    'stop_mult': stop_mult
                }
            }
        }
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
    best_pnl = results_df.nlargest(5, 'total_pnl')
    best_sharpe = results_df.nlargest(5, 'sharpe_ratio')
    best_balance = results_df.nlargest(5, 'balance_score')
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'position_optimization_{timestamp}.csv', index=False)
    
    # Print summary
    print("\nTop 5 by Total PnL:")
    summary_columns = ['regime_params', 'total_pnl', 'total_return', 'sharpe_ratio', 'win_rate', 'balance_score']
    print(best_pnl[summary_columns].to_string())
    
    print("\nTop 5 by Sharpe Ratio:")
    print(best_sharpe[summary_columns].to_string())
    
    print("\nTop 5 by Regime Balance:")
    print(best_balance[summary_columns].to_string())
    
    # Find best overall configuration
    # Weight metrics: 40% Sharpe, 30% PnL, 30% Balance
    results_df['normalized_pnl'] = results_df['total_pnl'] / results_df['total_pnl'].max()
    results_df['normalized_sharpe'] = results_df['sharpe_ratio'] / results_df['sharpe_ratio'].max()
    results_df['score'] = (
        0.4 * results_df['normalized_sharpe'] +
        0.3 * results_df['normalized_pnl'] +
        0.3 * results_df['balance_score']
    )
    
    best_config = results_df.nlargest(1, 'score').iloc[0]
    
    print("\nBest Overall Configuration:")
    print(f"Bear Position Scale: {best_config['regime_params']['bear_high_vol']['position_scale']:.2f}x")
    print(f"Bull Position Scale: {best_config['regime_params']['bull_high_vol']['position_scale']:.2f}x")
    print(f"Bear Reward/Risk: {best_config['regime_params']['bear_high_vol']['reward_risk']:.1f}")
    print(f"Bull Reward/Risk: {best_config['regime_params']['bull_high_vol']['reward_risk']:.1f}")
    print(f"Stop Multiplier: {best_config['regime_params']['bear_high_vol']['stop_mult']:.1f}")
    print(f"Total Return: {best_config['total_return']:.1%}")
    print(f"Total PnL: ${best_config['total_pnl']:,.2f}")
    print(f"Sharpe Ratio: {best_config['sharpe_ratio']:.2f}")
    print(f"Win Rate: {best_config['win_rate']:.1%}")
    print(f"Balance Score: {best_config['balance_score']:.2f}")
    print(f"Total Trades: {best_config['total_trades']}")
    
    # Save best parameters
    with open(f'best_position_params_{timestamp}.txt', 'w') as f:
        f.write("Best Parameters:\n\n")
        for key, value in best_config.items():
            if key not in ['regime_contribution']:
                f.write(f"{key}: {value}\n")
    
    print(f"\nFull results saved to: position_optimization_{timestamp}.csv")
    print(f"Best parameters saved to: best_position_params_{timestamp}.txt")
    
    return results_df

if __name__ == "__main__":
    results = run_position_optimization()
