import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
from strategy.vwap_obv_strategy import VWAPOBVCrossover

def load_data():
    """Load market data for both JNJ and XLV."""
    cache_file = 'data/market_data.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError("Market data not found. Run analyze_xlv_correlation.py first.")
    
    data = pd.read_pickle(cache_file)
    return data['JNJ'], data['XLV']

def prepare_data(df):
    """Prepare data for strategy."""
    # Remove timezone info and resample to hourly
    df.index = df.index.tz_localize(None)
    df = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
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
    yearly_pnl = daily_pnl.resample('Y').sum()
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

def run_strategy(config_path='final_strategy_params.json'):
    """Run the strategy with given configuration."""
    # Load configuration from JSON
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Loading data...")
    jnj_data, xlv_data = load_data()
    
    print("Preparing data...")
    jnj_df = prepare_data(jnj_data)
    xlv_df = prepare_data(xlv_data)
    
    print("\nInitializing strategy...")
    print("Configuration:")
    print("-" * 30)
    for key, value in config.items():
        if key != 'regime_params':
            print(f"{key}: {value}")
    print("\nRegime Parameters:")
    for regime, params in config['regime_params'].items():
        print(f"\n{regime}:")
        for param, value in params.items():
            print(f"  {param}: {value}")
    
    strategy = VWAPOBVCrossover(config)
    
    print("\nRunning strategy...")
    trades = strategy.run(jnj_df, xlv_df)
    
    if trades:
        trades_df = pd.DataFrame(trades)
        metrics = calculate_metrics(trades_df)
        
        if metrics:
            print("\nStrategy Performance:")
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
        plt.plot(jnj_df.index, jnj_df['close'], label='JNJ', alpha=0.7)
        
        buys = trades_df[trades_df['action'] == 'BUY']
        plt.scatter(buys['timestamp'], buys['price'], 
                   marker='^', color='g', label='Buy', s=100)
        
        sells = trades_df[trades_df['action'] == 'SELL']
        plt.scatter(sells['timestamp'], sells['price'],
                   marker='v', color='r', label='Sell', s=100)
        
        plt.title('JNJ Price and Trades')
        plt.legend()
        plt.grid(True)
        plt.savefig('strategy_results.png')
        plt.close()
        
        print("\nResults plotted to 'strategy_results.png'")
        trades_df.to_csv('strategy_trades.csv')
        print("Trade history saved to 'strategy_trades.csv'")
        
        return trades_df, metrics
    else:
        print("\nNo trades generated during the period.")
        return None, None

if __name__ == "__main__":
    trades_df, metrics = run_strategy()
