import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import pytz
from vwap_obv_strategy import VWAPOBVCrossover
import sys
sys.path.append('..')  # Add parent directory to path
from download_data import ensure_data_downloaded

def load_data(symbol: str):
    """Load market data for a symbol."""
    cache_file = f'../data/cache/{symbol.lower()}_data.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Market data not found for {symbol}")
    
    data = pd.read_pickle(cache_file)
    
    # Convert UTC timestamps to naive timestamps
    data.index = data.index.tz_localize(None)
    
    # Filter to 2022-2024
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    return data

def prepare_data(df):
    """Prepare data for strategy."""
    # Data is already hourly from Alpaca
    return df

def calculate_metrics(trades_df, capital=100000):
    """Calculate detailed performance metrics."""
    if trades_df.empty:
        return None
    
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
    
    # Calculate daily returns
    daily_returns = daily_pnl / capital
    
    # Calculate YoY returns
    yearly_pnl = daily_pnl.resample('YE').sum()  # Use YE instead of Y
    yearly_returns = yearly_pnl / capital
    
    # Calculate annualized return
    total_days = (daily_returns.index[-1] - daily_returns.index[0]).days
    total_years = total_days / 365
    total_return = total_pnl / capital
    annual_return = (1 + total_return) ** (1/total_years) - 1 if total_years > 0 else 0
    
    # Calculate Sharpe ratio (annualized, assuming 0% risk-free rate)
    daily_std = daily_returns.std()
    daily_mean = daily_returns.mean()
    sharpe_ratio = np.sqrt(252) * (daily_mean / daily_std) if daily_std > 0 else 0
    
    # Calculate max drawdown
    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    
    # Calculate regime-specific metrics
    regime_metrics = {}
    for regime in sells['regime'].unique():
        regime_trades = sells[sells['regime'] == regime]
        regime_pnl = regime_trades['pnl'].sum()
        regime_wins = len(regime_trades[regime_trades['pnl'] > 0])
        regime_metrics[regime] = {
            'trades': len(regime_trades),
            'win_rate': regime_wins / len(regime_trades),
            'total_pnl': regime_pnl,
            'avg_pnl': regime_trades['pnl'].mean(),
            'contribution': regime_pnl / total_pnl if total_pnl != 0 else 0
        }
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'yearly_returns': yearly_returns,
        'regime_metrics': regime_metrics
    }

def run_strategy(symbol: str, config_path='final_strategy_params.json'):
    """Run the strategy on a symbol."""
    # Load configuration from JSON
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\nTesting strategy on {symbol}...")
    print("=" * 50)
    
    print(f"\nLoading data for {symbol}...")
    data = load_data(symbol)
    
    print("\nPreparing data...")
    df = prepare_data(data)
    
    # Calculate volatility stats
    returns = df['close'].pct_change()
    vol = returns.rolling(window=20).std() * np.sqrt(252)
    print("\nVolatility Statistics:")
    print("-" * 30)
    print(f"Mean: {vol.mean():.1%}")
    print(f"Median: {vol.median():.1%}")
    for p in [25, 50, 67, 75, 90]:
        print(f"{p}th percentile: {vol.quantile(p/100):.1%}")
    
    print("\nInitializing strategy...")
    strategy = VWAPOBVCrossover(config)
    
    print("\nRunning strategy...")
    trades = strategy.run(df)
    
    if trades:
        trades_df = pd.DataFrame(trades)
        metrics = calculate_metrics(trades_df)
        
        if metrics:
            print(f"\nStrategy Performance for {symbol}:")
            print("=" * 50)
            
            print(f"\nOverall Metrics:")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.2%}")
            print(f"Total PnL: ${metrics['total_pnl']:,.2f}")
            print(f"Average Trade PnL: ${metrics['avg_pnl']:,.2f}")
            print(f"Annualized Return: {metrics['annual_return']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
            
            print("\nYearly Returns:")
            for year, ret in metrics['yearly_returns'].items():
                print(f"{year.year}: {ret:.2%}")
            
            print("\nPerformance by Regime:")
            print("-" * 30)
            for regime, stats in metrics['regime_metrics'].items():
                print(f"\n{regime}:")
                print(f"  trades: {stats['trades']}")
                print(f"  win_rate: {stats['win_rate']:.2%}")
                print(f"  avg_pnl: ${stats['avg_pnl']:,.2f}")
                print(f"  total_pnl: ${stats['total_pnl']:,.2f}")
                print(f"  contribution: {stats['contribution']:.2%}")
        
        # Plot results
        plt.figure(figsize=(15, 10))
        plt.plot(df.index, df['close'], label=symbol, alpha=0.7)
        
        buys = trades_df[trades_df['action'] == 'BUY']
        plt.scatter(buys['timestamp'], buys['price'], 
                   marker='^', color='g', label='Buy', s=100)
        
        sells = trades_df[trades_df['action'] == 'SELL']
        plt.scatter(sells['timestamp'], sells['price'],
                   marker='v', color='r', label='Sell', s=100)
        
        plt.title(f'{symbol} Price and Trades')
        plt.legend()
        plt.grid(True)
        
        # Save plot in results directory
        plot_file = os.path.join('results', f'strategy_results_{symbol}.png')
        plt.savefig(plot_file)
        plt.close()
        
        # Save trades in results directory
        trades_file = os.path.join('results', f'strategy_trades_{symbol}.csv')
        trades_df.to_csv(trades_file)
        
        print(f"\nResults plotted to {plot_file}")
        print(f"Trade history saved to {trades_file}")
        
        return trades_df, metrics
    else:
        print("\nNo trades generated during the period.")
        return None, None

if __name__ == "__main__":
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Ensure we have data for all symbols
    ensure_data_downloaded(
        config['symbols'],
        config['alpaca']['api_key'],
        config['alpaca']['api_secret'],
        base_path='..'  # Go up one directory since we're in strategy/
    )
    
    # Run strategy on all symbols
    results = {}
    for symbol in config['symbols']:
        try:
            trades_df, metrics = run_strategy(symbol)
            if metrics:
                results[symbol] = {
                    'annual_return': metrics['annual_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'total_pnl': metrics['total_pnl'],
                    'win_rate': metrics['win_rate'],
                    'total_trades': metrics['total_trades']
                }
            print("\n" + "="*80 + "\n")  # Separator between symbols
        except Exception as e:
            print(f"Error testing {symbol}: {e}")
            print("\n" + "="*80 + "\n")
    
    # Save summary results
    if results:
        summary_df = pd.DataFrame(results).T
        summary_file = os.path.join('results', 'summary.csv')
        summary_df.to_csv(summary_file)
        print("\nSummary of Results:")
        print(summary_df)
        print(f"\nResults saved to {summary_file}")
