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
    
    # Calculate Sharpe ratio
    daily_returns = daily_pnl / capital
    sharpe_ratio = np.sqrt(252) * (daily_returns.mean() / daily_returns.std()) if len(daily_returns) > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
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
            
            # Calculate regime balance score (closer to 75/25 is better)
            bear_contrib = metrics['regime_contribution'].get('bear_high_vol', {}).get('contribution', 0)
            bull_contrib = metrics['regime_contribution'].get('bull_high_vol', {}).get('contribution', 0)
            balance_score = 1 - abs(0.75 - bear_contrib) - abs(0.25 - bull_contrib)
            
            result = {
                **params,
                'total_pnl': metrics['total_pnl'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'win_rate': metrics['win_rate'],
                'max_drawdown': metrics['max_drawdown'],
                'balance_score': balance_score,
                'regime_contribution': metrics['regime_contribution']
            }
            
            print(f"PnL: ${metrics['total_pnl']:,.2f}, Sharpe: {metrics['sharpe_ratio']:.2f}, Balance: {balance_score:.2f}")
            
            return result
        return None
    except Exception as e:
        print(f"Error testing parameters: {e}")
        return None

def run_regime_optimization():
    """Run optimization focused on regime parameters."""
    # Load and prepare data
    print("Loading data...")
    jnj_data, xlv_data = load_data()
    jnj_df = prepare_data(jnj_data)
    xlv_df = prepare_data(xlv_data)
    
    # Fixed parameters
    base_params = {
        'MFI Period': 9,
        'VWAP Window': 90,
        'ATR Period': 3,
        'Regime Window': 20,
        'Min Stop Dollars': 0.50,
        'Max Stop Dollars': 2.00,
        'Risk Per Trade': 0.01,
        'Max Hold Hours': 24
    }
    
    # Parameter grid for regime optimization
    bear_scales = [1.5, 1.75, 2.0]
    bull_scales = [0.75, 1.0, 1.25, 1.5]
    bear_rr = [2.0, 2.5, 3.0]
    bull_rr = [1.5, 2.0, 2.5]
    bear_mfi = [55, 60, 65]
    bull_mfi = [65, 70, 75]
    trailing_distances = [0.3, 0.5, 0.7]  # Fraction of target distance
    
    param_combinations = []
    for bs, bus, br, bur, bm, bum, td in product(
        bear_scales, bull_scales, bear_rr, bull_rr, bear_mfi, bull_mfi, trailing_distances
    ):
        regime_params = {
            'bear_high_vol': {
                'position_scale': bs,
                'reward_risk': br,
                'stop_mult': 1.2,
                'mfi_overbought': bm,
                'trailing_stop': True
            },
            'bull_high_vol': {
                'position_scale': bus,
                'reward_risk': bur,
                'stop_mult': 1.2,
                'mfi_overbought': bum,
                'trailing_stop': True
            },
            'bear_med_vol': {
                'position_scale': 0.0,
                'reward_risk': 2.0,
                'stop_mult': 1.2
            },
            'bull_med_vol': {
                'position_scale': 0.0,
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
        
        params = base_params.copy()
        params['regime_params'] = regime_params
        params['trailing_distance'] = td
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
    best_balance = results_df.nlargest(10, 'balance_score')
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'regime_optimization_{timestamp}.csv', index=False)
    
    # Print summary
    print("\nTop 10 by Total PnL:")
    print(best_pnl[['total_pnl', 'sharpe_ratio', 'balance_score', 'regime_params']].to_string())
    
    print("\nTop 10 by Sharpe Ratio:")
    print(best_sharpe[['total_pnl', 'sharpe_ratio', 'balance_score', 'regime_params']].to_string())
    
    print("\nTop 10 by Regime Balance:")
    print(best_balance[['total_pnl', 'sharpe_ratio', 'balance_score', 'regime_params']].to_string())
    
    # Find best overall configuration
    # Weight metrics: 40% Sharpe, 40% PnL, 20% Balance
    results_df['normalized_pnl'] = results_df['total_pnl'] / results_df['total_pnl'].max()
    results_df['normalized_sharpe'] = results_df['sharpe_ratio'] / results_df['sharpe_ratio'].max()
    results_df['score'] = (
        0.4 * results_df['normalized_sharpe'] +
        0.4 * results_df['normalized_pnl'] +
        0.2 * results_df['balance_score']
    )
    
    best_config = results_df.nlargest(1, 'score').iloc[0]
    
    print("\nBest Overall Configuration:")
    print(f"Total PnL: ${best_config['total_pnl']:,.2f}")
    print(f"Sharpe Ratio: {best_config['sharpe_ratio']:.2f}")
    print(f"Balance Score: {best_config['balance_score']:.2f}")
    print("\nRegime Parameters:")
    for regime, params in best_config['regime_params'].items():
        if 'high_vol' in regime:
            print(f"\n{regime}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
    
    # Save best parameters
    with open(f'best_regime_params_{timestamp}.txt', 'w') as f:
        f.write("Best Parameters:\n\n")
        for key, value in best_config.items():
            if key != 'regime_contribution':
                f.write(f"{key}: {value}\n")
    
    print(f"\nFull results saved to: regime_optimization_{timestamp}.csv")
    print(f"Best parameters saved to: best_regime_params_{timestamp}.txt")
    
    return results_df

if __name__ == "__main__":
    results = run_regime_optimization()
