import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def analyze_trades(symbol: str):
    """Analyze trading patterns for a symbol."""
    trades_file = os.path.join('results', f'strategy_trades_{symbol}.csv')
    if not os.path.exists(trades_file):
        print(f"No trades found for {symbol}")
        return None
    
    # Load trades
    trades = pd.read_csv(trades_file)
    trades['timestamp'] = pd.to_datetime(trades['timestamp'])
    
    # Pair buys and sells
    buys = trades[trades['action'] == 'BUY'].copy()
    sells = trades[trades['action'] == 'SELL'].copy()
    
    # Calculate trade durations
    buys['duration'] = sells['timestamp'].values - buys['timestamp'].values
    buys['duration_hours'] = buys['duration'].dt.total_seconds() / 3600
    
    # Separate winners and losers
    winners = buys[sells['pnl'].values > 0]
    losers = buys[sells['pnl'].values <= 0]
    
    # Calculate win metrics
    win_rate = len(winners) / len(buys)
    avg_win = sells[sells['pnl'] > 0]['pnl'].mean()
    avg_loss = sells[sells['pnl'] <= 0]['pnl'].mean()
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    print(f"\nTrade Analysis for {symbol}")
    print("=" * 50)
    
    print("\nBasic Metrics:")
    print(f"Total Trades: {len(buys)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Average Win: ${avg_win:,.2f}")
    print(f"Average Loss: ${avg_loss:,.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    print("\nTrade Duration Statistics:")
    print("Winners:")
    print(f"  Mean: {winners['duration_hours'].mean():.1f} hours")
    print(f"  Median: {winners['duration_hours'].median():.1f} hours")
    print(f"  Max: {winners['duration_hours'].max():.1f} hours")
    print("\nLosers:")
    print(f"  Mean: {losers['duration_hours'].mean():.1f} hours")
    print(f"  Median: {losers['duration_hours'].median():.1f} hours")
    print(f"  Max: {losers['duration_hours'].max():.1f} hours")
    
    # Analyze win rate by regime
    print("\nWin Rate by Regime:")
    for regime in trades['regime'].unique():
        regime_sells = sells[sells['regime'] == regime]
        regime_wins = len(regime_sells[regime_sells['pnl'] > 0])
        regime_rate = regime_wins / len(regime_sells)
        regime_pnl = regime_sells['pnl'].sum()
        print(f"{regime}:")
        print(f"  Win Rate: {regime_rate:.2%}")
        print(f"  Total PnL: ${regime_pnl:,.2f}")
        print(f"  Avg PnL: ${regime_sells['pnl'].mean():,.2f}")
    
    # Plot trade PnL distribution
    plt.figure(figsize=(12, 6))
    plt.hist(sells['pnl'], bins=50, alpha=0.75)
    plt.title(f'{symbol} Trade PnL Distribution')
    plt.xlabel('PnL ($)')
    plt.ylabel('Number of Trades')
    plt.grid(True)
    plt.savefig(os.path.join('results', f'pnl_distribution_{symbol}.png'))
    plt.close()
    
    # Plot trade durations
    plt.figure(figsize=(12, 6))
    plt.hist(winners['duration_hours'], bins=30, alpha=0.5, label='Winners', color='green')
    plt.hist(losers['duration_hours'], bins=30, alpha=0.5, label='Losers', color='red')
    plt.title(f'{symbol} Trade Duration Distribution')
    plt.xlabel('Duration (hours)')
    plt.ylabel('Number of Trades')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join('results', f'duration_distribution_{symbol}.png'))
    plt.close()
    
    return {
        'symbol': symbol,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_winner_duration': winners['duration_hours'].mean(),
        'avg_loser_duration': losers['duration_hours'].mean()
    }

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Analyze all symbols
    symbols = ['META', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'COIN', 'JNJ']
    results = []
    
    for symbol in symbols:
        try:
            result = analyze_trades(symbol)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)
        df = df.set_index('symbol')
        
        # Calculate potential improvements
        df['risk_reward_ratio'] = abs(df['avg_win'] / df['avg_loss'])
        df['expected_value'] = df['win_rate'] * df['avg_win'] + (1 - df['win_rate']) * df['avg_loss']
        df['kelly_fraction'] = (df['win_rate'] - ((1 - df['win_rate']) / (df['avg_win'] / abs(df['avg_loss'])))) if any(df['avg_loss'] != 0) else 1
        
        print("\nPotential Improvements Analysis:")
        print("=" * 50)
        print("\nCurrent Metrics:")
        print(df[['win_rate', 'profit_factor', 'avg_win', 'avg_loss']].round(2))
        
        print("\nSuggested Position Sizing:")
        for symbol in df.index:
            kelly = df.loc[symbol, 'kelly_fraction']
            win_rate = df.loc[symbol, 'win_rate']
            profit_factor = df.loc[symbol, 'profit_factor']
            
            print(f"\n{symbol}:")
            print(f"Win Rate: {win_rate:.2%}")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Kelly Fraction: {kelly:.2%}")
            print(f"Suggested Position Size: {min(kelly, 0.5):.2%} of capital")  # Cap at 50%
            
            # Calculate potential return with optimal sizing
            current_ev = df.loc[symbol, 'expected_value']
            optimal_ev = current_ev * min(kelly, 0.5)
            print(f"Current Expected Value per Trade: ${current_ev:,.2f}")
            print(f"Optimal Expected Value per Trade: ${optimal_ev:,.2f}")
