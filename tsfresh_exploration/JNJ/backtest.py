import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from strategy.vwap_obv_strategy import VWAPOBVCrossover

def load_data():
    """Load cached market data."""
    cache_file = 'data/market_data.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError("Market data not found. Run analyze_xlv_correlation.py first.")
    
    data = pd.read_pickle(cache_file)
    return data['JNJ']

def prepare_data(df):
    """Prepare data for backtesting."""
    # Remove timezone info and resample to hourly
    df.index = df.index.tz_localize(None)
    df = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return df

def calculate_metrics(signals, data):
    """Calculate trading performance metrics."""
    if not signals:
        return {}
    
    trades = []
    current_trade = None
    
    for signal in signals:
        if signal['action'] == 'BUY':
            current_trade = signal
        elif signal['action'] == 'SELL' and current_trade:
            trade = {
                'entry_time': current_trade['timestamp'],
                'exit_time': signal['timestamp'],
                'entry_price': current_trade['price'],
                'exit_price': signal['price'],
                'size': current_trade['size'],
                'pnl': signal['pnl'],
                'duration': (signal['timestamp'] - current_trade['timestamp']).total_seconds() / 3600
            }
            trades.append(trade)
            current_trade = None
    
    if not trades:
        return {}
    
    # Convert to DataFrame for analysis
    trades_df = pd.DataFrame(trades)
    
    # Calculate metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
    profit_factor = avg_win / avg_loss if avg_loss != 0 else float('inf')
    
    # Calculate drawdown
    cumulative_pnl = trades_df['pnl'].cumsum()
    rolling_max = cumulative_pnl.expanding().max()
    drawdowns = (cumulative_pnl - rolling_max) / rolling_max * 100
    max_drawdown = abs(drawdowns.min())
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    returns = trades_df['pnl'] / trades_df['size'] / trades_df['entry_price']
    sharpe = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
    
    return {
        'Total Trades': total_trades,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Max Drawdown %': max_drawdown,
        'Sharpe Ratio': sharpe,
        'Average Duration (hours)': trades_df['duration'].mean(),
        'Total PnL': trades_df['pnl'].sum(),
        'PnL per Trade': trades_df['pnl'].mean()
    }

def plot_results(trades_df, data, title="Strategy Performance"):
    """Plot backtest results."""
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Equity curve
    plt.subplot(2, 1, 1)
    cumulative_pnl = trades_df['pnl'].cumsum()
    plt.plot(cumulative_pnl.index, cumulative_pnl.values)
    plt.title('Equity Curve')
    plt.grid(True)
    
    # Plot 2: Trade distribution
    plt.subplot(2, 2, 3)
    trades_df['pnl'].hist(bins=50)
    plt.title('PnL Distribution')
    plt.grid(True)
    
    # Plot 3: Drawdown
    plt.subplot(2, 2, 4)
    rolling_max = cumulative_pnl.expanding().max()
    drawdown = (cumulative_pnl - rolling_max) / rolling_max * 100
    plt.plot(drawdown.index, drawdown.values)
    plt.title('Drawdown (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')

def run_backtest(config):
    """Run backtest with given configuration."""
    print("Loading data...")
    data = load_data()
    df = prepare_data(data)
    
    print("Initializing strategy...")
    strategy = VWAPOBVCrossover(config)
    
    print("Generating signals...")
    signals = strategy.generate_signals(df, capital=100000)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(signals, df)
    
    print("\nBacktest Results:")
    print("=" * 50)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.2f}")
        else:
            print(f"{metric}: {value}")
    
    if signals:
        trades_df = pd.DataFrame(signals)
        plot_results(trades_df, df)
        print("\nPlots saved as 'backtest_results.png'")
    
    return metrics, signals

def optimize_parameters():
    """Find optimal strategy parameters."""
    param_grid = {
        'Volume Pattern Threshold': [1.5, 2.0, 2.5],
        'Position Scale Max': [1.25, 1.5, 1.75],
    }
    
    results = []
    
    for threshold in param_grid['Volume Pattern Threshold']:
        for scale_max in param_grid['Position Scale Max']:
            config = {
                'Candle Size': 60,
                'Strategy Mode': 'INTRADAY',
                'Volume Pattern Threshold': threshold,
                'Position Scale Max': scale_max
            }
            
            print(f"\nTesting configuration:")
            print(f"Threshold: {threshold}, Scale Max: {scale_max}")
            
            metrics, _ = run_backtest(config)
            metrics['Volume Pattern Threshold'] = threshold
            metrics['Position Scale Max'] = scale_max
            results.append(metrics)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find best configuration based on Sharpe ratio
    best_config = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
    
    print("\nOptimal Configuration:")
    print("=" * 50)
    print(f"Volume Pattern Threshold: {best_config['Volume Pattern Threshold']}")
    print(f"Position Scale Max: {best_config['Position Scale Max']}")
    print(f"Sharpe Ratio: {best_config['Sharpe Ratio']:.2f}")
    print(f"Win Rate: {best_config['Win Rate']:.2%}")
    print(f"Profit Factor: {best_config['Profit Factor']:.2f}")
    
    return results_df

if __name__ == "__main__":
    # Run parameter optimization
    print("Running parameter optimization...")
    results = optimize_parameters()
    
    # Save results
    results.to_csv('optimization_results.csv')
    print("\nOptimization results saved to 'optimization_results.csv'")
