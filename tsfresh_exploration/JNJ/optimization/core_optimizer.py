import os
import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from strategy.vwap_obv_strategy import VWAPOBVCrossover
from run_strategy import load_data, prepare_data
import multiprocessing as mp
from functools import partial

def validate_trade_pnl(trades_df):
    """Validate individual trade PnLs."""
    sells = trades_df[trades_df['action'] == 'SELL']
    buys = trades_df[trades_df['action'] == 'BUY']
    
    validation = []
    for i, sell in sells.iterrows():
        # Find corresponding buy
        buy = buys.iloc[len(validation)]
        
        # Calculate PnL
        price_diff = sell['price'] - buy['price']
        size = sell['size']
        expected_pnl = price_diff * size
        actual_pnl = sell['pnl']
        
        # Validate
        if abs(expected_pnl - actual_pnl) > 0.01:  # Allow for small float differences
            print(f"\nPnL Validation Error:")
            print(f"Trade {len(validation)+1}:")
            print(f"Buy: ${buy['price']:.2f} x {buy['size']} shares")
            print(f"Sell: ${sell['price']:.2f}")
            print(f"Expected PnL: ${expected_pnl:.2f}")
            print(f"Actual PnL: ${actual_pnl:.2f}")
        
        validation.append({
            'entry_price': buy['price'],
            'exit_price': sell['price'],
            'size': size,
            'pnl': actual_pnl,
            'regime': sell['regime']
        })
    
    return pd.DataFrame(validation)

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
    
    # Validate trades
    validated_trades = validate_trade_pnl(trades_df)
    
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
    for regime in validated_trades['regime'].unique():
        regime_trades = validated_trades[validated_trades['regime'] == regime]
        regime_pnl = regime_trades['pnl'].sum()
        regime_stats[regime] = {
            'trades': len(regime_trades),
            'win_rate': len(regime_trades[regime_trades['pnl'] > 0]) / len(regime_trades),
            'avg_pnl': regime_trades['pnl'].mean(),
            'total_pnl': regime_pnl,
            'contribution': regime_pnl / total_pnl if total_pnl != 0 else 0
        }
    
    # Print detailed metrics
    print("\nDetailed Performance Metrics:")
    print(f"Starting Capital: ${capital:,.2f}")
    print(f"Ending Capital: ${(capital * (1 + total_return)):,.2f}")
    print(f"Total Return: {total_return:.1%}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Trade: ${avg_pnl:,.2f}")
    print(f"Max Drawdown: {max_drawdown:.1%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print("\nRegime Performance:")
    for regime, stats in regime_stats.items():
        print(f"\n{regime}:")
        print(f"  Trades: {stats['trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1%}")
        print(f"  Avg PnL: ${stats['avg_pnl']:,.2f}")
        print(f"  Total PnL: ${stats['total_pnl']:,.2f}")
        print(f"  Contribution: {stats['contribution']:.1%}")
    
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
            
            # Calculate risk-adjusted score
            win_rate_score = metrics['win_rate']
            pnl_score = min(metrics['total_pnl'] / 100000, 1)  # Cap at 100% return
            drawdown_score = max(0, 1 + metrics['max_drawdown'])  # Convert to positive score
            
            risk_adjusted_score = (
                0.4 * win_rate_score +
                0.4 * pnl_score +
                0.2 * drawdown_score
            )
            
            result = {
                **params,
                'total_pnl': metrics['total_pnl'],
                'total_return': metrics['total_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'win_rate': metrics['win_rate'],
                'max_drawdown': metrics['max_drawdown'],
                'total_trades': metrics['total_trades'],
                'avg_trade_pnl': metrics['avg_pnl'],
                'risk_adjusted_score': risk_adjusted_score,
                'regime_contribution': metrics['regime_contribution']
            }
            
            print(f"\nParameters:")
            print(f"Min Stop: ${params['Min Stop Dollars']:.2f}")
            print(f"Max Stop: ${params['Max Stop Dollars']:.2f}")
            print(f"Risk: {params['Risk Per Trade']:.1%}")
            print(f"Hold: {params['Max Hold Hours']}h")
            print(f"Return: {metrics['total_return']:.1%}")
            print(f"PnL: ${metrics['total_pnl']:,.2f}")
            print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"Win Rate: {metrics['win_rate']:.1%}")
            print("---")
            
            return result
        return None
    except Exception as e:
        print(f"Error testing parameters: {e}")
        return None

def run_core_optimization():
    """Run optimization focused on risk parameters."""
    # Load and prepare data
    print("Loading data...")
    jnj_data, xlv_data = load_data()
    jnj_df = prepare_data(jnj_data)
    xlv_df = prepare_data(xlv_data)
    
    # Parameter grid for risk parameters
    param_grid = {
        'Min Stop Dollars': [0.25, 0.50, 0.75, 1.00, 1.25],          # Test higher stops
        'Max Stop Dollars': [1.50, 2.00, 2.50, 3.00, 3.50],         # Test higher caps
        'Risk Per Trade': [0.005, 0.01, 0.015, 0.02, 0.025],        # Test higher risk
        'Max Hold Hours': [12, 24, 36, 48, 72]                       # Test longer holds
    }
    
    # Fixed parameters (using optimized values)
    fixed_params = {
        'VWAP Window': 50,                    # Optimized
        'MFI Period': 9,                      # Optimized
        'Regime Window': 20,                  # Optimized
        'ATR Period': 2,                      # Optimized
        'mfi_entry': 30,                      # Optimized
        'bear_exit': 55,                      # Optimized
        'bull_exit': 75                       # Optimized
    }
    
    # Generate parameter combinations
    param_combinations = []
    for min_stop, max_stop, risk, hold in product(
        param_grid['Min Stop Dollars'],
        param_grid['Max Stop Dollars'],
        param_grid['Risk Per Trade'],
        param_grid['Max Hold Hours']
    ):
        # Skip invalid combinations
        if min_stop >= max_stop:
            continue
            
        params = {
            **fixed_params,
            'Min Stop Dollars': min_stop,
            'Max Stop Dollars': max_stop,
            'Risk Per Trade': risk,
            'Max Hold Hours': hold,
            'regime_params': {
                'bear_high_vol': {
                    'position_scale': 1.75,
                    'reward_risk': 2.5,
                    'stop_mult': 1.2,
                    'mfi_overbought': 55,     # Optimized
                    'trailing_stop': True
                },
                'bull_high_vol': {
                    'position_scale': 1.0,
                    'reward_risk': 2.0,
                    'stop_mult': 1.2,
                    'mfi_overbought': 75,     # Optimized
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
    best_risk = results_df.nlargest(5, 'risk_adjusted_score')
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_csv(f'risk_optimization_{timestamp}.csv', index=False)
    
    # Print summary
    print("\nTop 5 by Total PnL:")
    summary_columns = ['Min Stop Dollars', 'Max Stop Dollars', 'Risk Per Trade', 'Max Hold Hours', 
                      'total_pnl', 'total_return', 'sharpe_ratio', 'win_rate']
    print(best_pnl[summary_columns].to_string())
    
    print("\nTop 5 by Sharpe Ratio:")
    print(best_sharpe[summary_columns].to_string())
    
    print("\nTop 5 by Risk-Adjusted Score:")
    print(best_risk[summary_columns].to_string())
    
    # Find best overall configuration
    # Weight metrics: 40% Risk-Adjusted, 40% Sharpe, 20% PnL
    results_df['normalized_pnl'] = results_df['total_pnl'] / results_df['total_pnl'].max()
    results_df['normalized_sharpe'] = results_df['sharpe_ratio'] / results_df['sharpe_ratio'].max()
    results_df['score'] = (
        0.4 * results_df['risk_adjusted_score'] +
        0.4 * results_df['normalized_sharpe'] +
        0.2 * results_df['normalized_pnl']
    )
    
    best_config = results_df.nlargest(1, 'score').iloc[0]
    
    print("\nBest Overall Configuration:")
    print(f"Min Stop: ${best_config['Min Stop Dollars']:.2f}")
    print(f"Max Stop: ${best_config['Max Stop Dollars']:.2f}")
    print(f"Risk Per Trade: {best_config['Risk Per Trade']:.1%}")
    print(f"Max Hold Hours: {best_config['Max Hold Hours']}")
    print(f"Total Return: {best_config['total_return']:.1%}")
    print(f"Total PnL: ${best_config['total_pnl']:,.2f}")
    print(f"Sharpe Ratio: {best_config['sharpe_ratio']:.2f}")
    print(f"Win Rate: {best_config['win_rate']:.1%}")
    print(f"Total Trades: {best_config['total_trades']}")
    
    # Save best parameters
    with open(f'best_risk_params_{timestamp}.txt', 'w') as f:
        f.write("Best Parameters:\n\n")
        for key, value in best_config.items():
            if key not in ['regime_contribution', 'regime_params']:
                f.write(f"{key}: {value}\n")
    
    print(f"\nFull results saved to: risk_optimization_{timestamp}.csv")
    print(f"Best parameters saved to: best_risk_params_{timestamp}.txt")
    
    return results_df

if __name__ == "__main__":
    results = run_core_optimization()
