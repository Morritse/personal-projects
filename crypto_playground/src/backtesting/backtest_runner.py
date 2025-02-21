import asyncio
from optimizer import StrategyOptimizer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime

async def run_optimization(
    symbols=["BTC/USD", "ETH/USD"],
    lookback_days=30,
    max_combinations=100
):
    """Run parameter optimization and plot results."""
    # Create optimizer
    optimizer = StrategyOptimizer(
        symbols=symbols,
        lookback_days=lookback_days
    )
    
    # Run optimization
    results = await optimizer.optimize(max_combinations)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Convert results to DataFrame for analysis
    if not results:
        print("No valid results found. Try adjusting parameters.")
        return results
        
    df = pd.DataFrame([{
        'sharpe_ratio': r.sharpe_ratio,
        'profit_loss': r.profit_loss,
        'win_rate': r.win_rate,
        'max_drawdown': r.max_drawdown,
        'total_trades': r.total_trades,
        'avg_duration': r.avg_trade_duration,
        **r.params
    } for r in results])
    
    # Create visualization plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.style.use('default')  # Use default matplotlib style
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Parameter Optimization Results', fontsize=16)
    
    # 1. Sharpe Ratio vs Profit/Loss scatter
    ax = axes[0, 0]
    sns.scatterplot(
        data=df,
        x='sharpe_ratio',
        y='profit_loss',
        size='total_trades',
        hue='win_rate',
        ax=ax
    )
    ax.set_title('Sharpe Ratio vs Profit/Loss')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Profit/Loss %')
    
    # 2. Parameter distributions for top performers
    ax = axes[0, 1]
    if len(df) > 0:  # Only plot if we have data
        top_performers = df[df['sharpe_ratio'] > df['sharpe_ratio'].quantile(0.8)]
        if len(top_performers) > 0:
            param_counts = {}
            for param in ['MIN_SIGNAL_STRENGTH', 'MIN_CONFIDENCE', 'STOP_LOSS_ATR', 'TAKE_PROFIT_ATR']:
                param_counts[param] = top_performers[param].value_counts()
            
            param_df = pd.DataFrame(param_counts)
            if not param_df.empty:
                param_df.plot(kind='bar', ax=ax)
                ax.set_title('Parameter Distribution (Top 20%)')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            else:
                ax.text(0.5, 0.5, 'No parameter distribution data available',
                       ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'No top performers found',
                   ha='center', va='center')
    else:
        ax.text(0.5, 0.5, 'No optimization results available',
               ha='center', va='center')
    
    # 3. Win Rate vs Max Drawdown
    ax = axes[1, 0]
    sns.scatterplot(
        data=df,
        x='win_rate',
        y='max_drawdown',
        size='total_trades',
        hue='sharpe_ratio',
        ax=ax
    )
    ax.set_title('Win Rate vs Max Drawdown')
    ax.set_xlabel('Win Rate %')
    ax.set_ylabel('Max Drawdown %')
    
    # 4. Trade frequency vs Duration
    ax = axes[1, 1]
    sns.scatterplot(
        data=df,
        x='total_trades',
        y='avg_duration',
        size='profit_loss',
        hue='sharpe_ratio',
        ax=ax
    )
    ax.set_title('Trade Frequency vs Duration')
    ax.set_xlabel('Number of Trades')
    ax.set_ylabel('Average Trade Duration (min)')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'results/optimization_plots_{timestamp}.png')
    
    # Print summary statistics
    print("\nOptimization Results Summary:")
    print("-" * 50)
    print(f"Total Parameter Combinations Tested: {len(df)}")
    print(f"\nBest Sharpe Ratio: {df['sharpe_ratio'].max():.2f}")
    print(f"Best Profit/Loss: {df['profit_loss'].max():.2%}")
    print(f"Best Win Rate: {df['win_rate'].max():.2%}")
    print(f"\nParameter Ranges Found in Top 20% of Results:")
    
    # Get parameter columns (exclude metric columns)
    param_columns = [col for col in df.columns if col not in [
        'sharpe_ratio', 'profit_loss', 'win_rate', 'max_drawdown', 
        'total_trades', 'avg_duration'
    ]]
    
    # Calculate ranges for each parameter from top performers
    for param in param_columns:
        param_values = top_performers[param].dropna()  # Remove any NaN values
        if not param_values.empty:
            param_min = param_values.min()
            param_max = param_values.max()
            param_mean = param_values.mean()
            param_std = param_values.std()
            
            print(f"\n{param}:")
            print(f"  Range: {param_min:.2f} to {param_max:.2f}")
            print(f"  Mean: {param_mean:.2f} ± {param_std:.2f}")
        else:
            print(f"\n{param}: No valid values in top performers")
    
    return results

if __name__ == '__main__':
    # Run optimization with default settings
    asyncio.run(run_optimization())
