import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def process_market_data(data):
    """Process raw market data into analysis-ready format."""
    # Create base DataFrame with JNJ data
    df = data['JNJ'].copy()
    df.index = df.index.tz_localize(None)  # Remove timezone info
    df = df.resample('h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).ffill()
    
    # Add XLV data
    xlv_data = data['XLV'].copy()
    xlv_data.index = xlv_data.index.tz_localize(None)
    xlv_data = xlv_data.resample('h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).ffill()
    
    # Rename columns
    df = df.rename(columns={
        'close': 'jnj_close',
        'volume': 'jnj_volume'
    })
    df['xlv_close'] = xlv_data['close']
    df['xlv_volume'] = xlv_data['volume']
    
    # Calculate returns (using log returns for better statistical properties)
    df['jnj_returns_1H'] = np.log(df['jnj_close'] / df['jnj_close'].shift(1))
    df['xlv_returns_1H'] = np.log(df['xlv_close'] / df['xlv_close'].shift(1))
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    # Only keep rows during market hours (9:30-16:00 ET) and after hours (16:00-23:00 ET)
    df = df[
        ((df.index.hour >= 9) & (df.index.hour < 16)) |
        ((df.index.hour >= 16) & (df.index.hour <= 23))
    ]
    
    return df

def classify_regimes(df, window=20):
    """Classify market regimes based on XLV returns and volatility."""
    # Calculate rolling returns and volatility for XLV
    xlv_ret = df['xlv_returns_1H'].rolling(window=window, min_periods=window//2).mean() * 252  # Annualized
    xlv_vol = df['xlv_returns_1H'].rolling(window=window, min_periods=window//2).std() * np.sqrt(252)  # Annualized
    
    # Calculate dynamic volatility threshold
    vol_threshold = xlv_vol.rolling(window=window*5, min_periods=window).median()
    
    # Define market regimes
    df['regime'] = 'normal'
    df.loc[(xlv_ret > 0) & (xlv_vol <= vol_threshold), 'regime'] = 'bull_low_vol'
    df.loc[(xlv_ret > 0) & (xlv_vol > vol_threshold), 'regime'] = 'bull_high_vol'
    df.loc[(xlv_ret <= 0) & (xlv_vol <= vol_threshold), 'regime'] = 'bear_low_vol'
    df.loc[(xlv_ret <= 0) & (xlv_vol > vol_threshold), 'regime'] = 'bear_high_vol'
    
    return df

def analyze_afterhours_opportunities(df):
    """Analyze after-hours mean reversion opportunities."""
    
    # Define after-hours period (16:00-23:00 ET)
    df['afterhours'] = (df.index.hour >= 16) & (df.index.hour <= 23)
    
    # Add trading time restrictions
    df['valid_entry'] = (
        df['afterhours'] &
        (df.index.minute >= 15)  # Allow for calculation warmup
    )
    
    # Calculate returns and spreads
    timeframes = {'2H': 2, '4H': 4}
    results = {}
    
    for label, periods in timeframes.items():
        # Calculate JNJ-XLV spread
        spread = df['jnj_returns_1H'] - df['xlv_returns_1H']
        
        # Calculate rolling stats for z-score
        rolling_mean = spread.rolling(window=20, min_periods=10).mean()
        rolling_std = spread.rolling(window=20, min_periods=10).std()
        zscore = (spread - rolling_mean) / rolling_std
        
        # Analyze divergences by regime
        regimes = ['bull_high_vol', 'bull_low_vol', 'bear_high_vol', 'bear_low_vol']
        regime_results = {}
        
        for regime in regimes:
            # Initialize empty results
            regime_results[regime] = {
                'success_rate': 0,
                'total_trades': 0,
                'avg_return': 0,
                'avg_duration': 0,
                'sharpe': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'returns': [],
                'trade_times': [],
                'max_win_streak': 0,
                'max_loss_streak': 0
            }
            
            # Find divergences
            regime_data = df[df['regime'] == regime].copy()
            regime_data['zscore'] = zscore
            divergences = regime_data[
                (regime_data['valid_entry']) & 
                (abs(regime_data['zscore']) > 2.0)
            ]
            
            if len(divergences) > 0:
                # Track convergence stats
                convergence_count = 0
                total_count = 0
                returns = []
                durations = []
                trade_times = []
                
                for timestamp in divergences.index:
                    # Look forward required periods
                    forward_mask = (df.index > timestamp) & (df.index <= timestamp + pd.Timedelta(hours=periods))
                    if not forward_mask.any():
                        continue
                    
                    forward_data = df[forward_mask]
                    if len(forward_data) < periods:
                        continue
                    
                    # Get initial and final values
                    initial_spread = spread[timestamp]
                    forward_spreads = spread[forward_data.index]
                    
                    # Check for convergence
                    converged = False
                    convergence_time = None
                    
                    for t, v in forward_spreads.items():
                        if abs(v) < abs(initial_spread):
                            converged = True
                            convergence_time = t
                            break
                    
                    if converged:
                        convergence_count += 1
                        # Calculate return
                        if zscore[timestamp] > 0:  # JNJ overvalued
                            trade_return = -(df['jnj_close'][convergence_time] / df['jnj_close'][timestamp] - 1)
                        else:  # JNJ undervalued
                            trade_return = df['jnj_close'][convergence_time] / df['jnj_close'][timestamp] - 1
                        returns.append(trade_return)
                        # Calculate duration
                        duration = (convergence_time - timestamp).total_seconds() / 3600  # hours
                        durations.append(duration)
                        # Store trade time
                        trade_times.append(timestamp.strftime('%H:%M'))
                    
                    total_count += 1
                
                if total_count > 0:
                    # Update trade statistics
                    wins = np.array(returns) > 0
                    regime_results[regime].update({
                        'success_rate': convergence_count / total_count,
                        'total_trades': total_count,
                        'avg_return': np.mean(returns) if returns else 0,
                        'avg_duration': np.mean(durations) if durations else 0,
                        'sharpe': np.mean(returns) / np.std(returns) if returns else 0,
                        'win_rate': np.mean(wins) if len(wins) > 0 else 0,
                        'avg_win': np.mean(np.array(returns)[wins]) if any(wins) else 0,
                        'avg_loss': np.mean(np.array(returns)[~wins]) if any(~wins) else 0,
                        'profit_factor': abs(np.sum(np.array(returns)[wins]) / np.sum(np.array(returns)[~wins])) if any(~wins) else np.inf,
                        'returns': returns,
                        'trade_times': trade_times
                    })
                    
                    # Calculate streaks
                    if len(wins) > 1:
                        streaks = []
                        current_streak = 1
                        for i in range(1, len(wins)):
                            if wins[i] == wins[i-1]:
                                current_streak += 1
                            else:
                                streaks.append(current_streak)
                                current_streak = 1
                        streaks.append(current_streak)
                        
                        regime_results[regime].update({
                            'max_win_streak': max([s for i, s in enumerate(streaks) if wins[sum(streaks[:i])]] or [0]),
                            'max_loss_streak': max([s for i, s in enumerate(streaks) if not wins[sum(streaks[:i])]] or [0])
                        })
        
        results[label] = regime_results
    
    return results

def plot_results(results):
    """Plot analysis results."""
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Success rates by regime and timeframe
    plt.subplot(3, 2, 1)
    regimes = ['bull_high_vol', 'bull_low_vol', 'bear_high_vol', 'bear_low_vol']
    x = np.arange(len(regimes))
    width = 0.35
    
    for i, (timeframe, regime_results) in enumerate(results.items()):
        rates = [regime_results[r]['success_rate'] * 100 for r in regimes]
        plt.bar(x + i*width, rates, width, label=timeframe)
    
    plt.xlabel('Market Regime')
    plt.ylabel('Success Rate (%)')
    plt.title('After-Hours Mean Reversion Success Rate')
    plt.xticks(x + width/2, [r.replace('_', ' ').title() for r in regimes], rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Average returns by regime
    plt.subplot(3, 2, 2)
    for i, (timeframe, regime_results) in enumerate(results.items()):
        returns = [regime_results[r]['avg_return'] * 100 for r in regimes]
        plt.bar(x + i*width, returns, width, label=timeframe)
    
    plt.xlabel('Market Regime')
    plt.ylabel('Average Return (%)')
    plt.title('Average Trade Return by Regime')
    plt.xticks(x + width/2, [r.replace('_', ' ').title() for r in regimes], rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Return distributions
    plt.subplot(3, 2, 3)
    for timeframe, regime_results in results.items():
        all_returns = []
        for regime in regimes:
            all_returns.extend(regime_results[regime]['returns'])
        if all_returns:
            plt.hist(np.array(all_returns) * 100, bins=50, alpha=0.5, label=timeframe)
    
    plt.xlabel('Return (%)')
    plt.ylabel('Frequency')
    plt.title('Return Distribution')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Sharpe ratio by regime
    plt.subplot(3, 2, 4)
    for i, (timeframe, regime_results) in enumerate(results.items()):
        sharpes = [regime_results[r]['sharpe'] for r in regimes]
        plt.bar(x + i*width, sharpes, width, label=timeframe)
    
    plt.xlabel('Market Regime')
    plt.ylabel('Sharpe Ratio')
    plt.title('Risk-Adjusted Performance by Regime')
    plt.xticks(x + width/2, [r.replace('_', ' ').title() for r in regimes], rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Win/Loss ratio by regime
    plt.subplot(3, 2, 5)
    for i, (timeframe, regime_results) in enumerate(results.items()):
        ratios = [abs(regime_results[r]['avg_win']/regime_results[r]['avg_loss']) 
                 if regime_results[r]['avg_loss'] != 0 else 0 
                 for r in regimes]
        plt.bar(x + i*width, ratios, width, label=timeframe)
    
    plt.xlabel('Market Regime')
    plt.ylabel('Win/Loss Ratio')
    plt.title('Win/Loss Ratio by Regime')
    plt.xticks(x + width/2, [r.replace('_', ' ').title() for r in regimes], rotation=45)
    plt.legend()
    plt.grid(True)
    
    # Plot 6: Trade timing distribution
    plt.subplot(3, 2, 6)
    all_times = []
    for timeframe, regime_results in results.items():
        for regime in regimes:
            all_times.extend(regime_results[regime].get('trade_times', []))
    
    if all_times:
        times = pd.to_datetime(all_times, format='%H:%M').hour
        plt.hist(times, bins=8, range=(16, 24), alpha=0.75)
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Trades')
        plt.title('Trade Entry Time Distribution')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('afterhours_analysis.png')

def print_results(results):
    """Print detailed analysis results."""
    print("\nAfter-Hours Mean Reversion Analysis")
    print("=" * 50)
    
    for timeframe, regime_results in results.items():
        print(f"\n{timeframe} Timeframe Results:")
        print("-" * 30)
        
        for regime, stats in regime_results.items():
            print(f"\n{regime.replace('_', ' ').title()}:")
            print(f"Success Rate: {stats['success_rate']:.2%}")
            print(f"Total Trades: {stats['total_trades']}")
            print(f"Win Rate: {stats['win_rate']:.2%}")
            print(f"Average Win: {stats['avg_win']:.2%}")
            print(f"Average Loss: {stats['avg_loss']:.2%}")
            print(f"Profit Factor: {stats['profit_factor']:.2f}")
            print(f"Max Win Streak: {stats['max_win_streak']}")
            print(f"Max Loss Streak: {stats['max_loss_streak']}")
            print(f"Average Duration: {stats['avg_duration']:.1f} hours")
            print(f"Sharpe Ratio: {stats['sharpe']:.2f}")

if __name__ == "__main__":
    # Get market data
    print("Loading market data...")
    cache_file = 'data/market_data.pkl'
    if not os.path.exists(cache_file):
        print("Please run analyze_xlv_correlation.py first to generate market data")
        exit(1)
    
    data = pd.read_pickle(cache_file)
    
    # Process data
    print("Processing data...")
    df = process_market_data(data)
    
    # Classify regimes
    print("Classifying market regimes...")
    df = classify_regimes(df)
    
    # Analyze after-hours opportunities
    print("\nAnalyzing after-hours opportunities...")
    results = analyze_afterhours_opportunities(df)
    
    # Generate visualizations
    plot_results(results)
    
    # Print detailed results
    print_results(results)
