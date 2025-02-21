import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def simulate_options_strategy(win_rate=0.60, num_trades=1000, strategy_type='long_options'):
    """Simulate different options strategies with our win rate."""
    np.random.seed(42)
    
    if strategy_type == 'long_options':
        # Long OTM options strategy
        # Risk $100 per trade to make $300-500 on winners
        risk_per_trade = 100
        min_reward = 300
        max_reward = 500
        
        # Generate random wins/losses
        is_win = np.random.random(num_trades) < win_rate
        
        # Calculate PnL
        rewards = np.random.uniform(min_reward, max_reward, num_trades)
        pnl = np.where(is_win, rewards, -risk_per_trade)
        
    elif strategy_type == 'spreads':
        # Credit/Debit spreads strategy
        # Risk $200 to make $100 but higher probability
        risk_per_trade = 200
        reward_per_trade = 100
        # Adjust win rate up for spreads
        win_rate = win_rate + 0.1  # Add 10% for defined risk
        
        # Generate random wins/losses
        is_win = np.random.random(num_trades) < win_rate
        
        # Calculate PnL
        pnl = np.where(is_win, reward_per_trade, -risk_per_trade)
        
    elif strategy_type == 'quick_scalps':
        # Quick ATM option scalps
        # Risk $200 per trade for quick $100-300 scalps
        risk_per_trade = 200
        min_reward = 100
        max_reward = 300
        
        # Generate random wins/losses
        is_win = np.random.random(num_trades) < win_rate
        
        # Calculate PnL
        rewards = np.random.uniform(min_reward, max_reward, num_trades)
        pnl = np.where(is_win, rewards, -risk_per_trade)
    
    # Calculate metrics
    cumulative_pnl = np.cumsum(pnl)
    total_pnl = cumulative_pnl[-1]
    win_count = np.sum(is_win)
    actual_win_rate = win_count / num_trades
    avg_win = np.mean(pnl[pnl > 0])
    avg_loss = np.mean(pnl[pnl < 0])
    profit_factor = abs(avg_win / avg_loss)
    
    # Calculate drawdown
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - running_max
    max_drawdown = np.min(drawdown)
    
    # Calculate daily returns for Sharpe
    trades_per_day = 4  # Assume 4 trades per day
    days = num_trades // trades_per_day
    daily_pnl = np.array_split(pnl, days)
    daily_returns = np.array([sum(day) for day in daily_pnl])
    sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) != 0 else 0
    
    # Calculate additional risk metrics
    win_loss_ratio = abs(avg_win / avg_loss)
    expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    return {
        'strategy': strategy_type,
        'total_pnl': total_pnl,
        'win_rate': actual_win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'cumulative_pnl': cumulative_pnl,
        'win_loss_ratio': win_loss_ratio,
        'expected_value': expected_value,
        'avg_trade': expected_value
    }

def analyze_options_strategies():
    """Analyze different options strategies using our win rate."""
    strategies = ['long_options', 'spreads', 'quick_scalps']
    results = []
    
    print("Simulating 1000 trades for each strategy with 60% base win rate\n")
    
    # Run simulations
    for strategy in strategies:
        result = simulate_options_strategy(strategy_type=strategy)
        results.append(result)
        
        print(f"\n{strategy.replace('_', ' ').title()} Strategy:")
        print("-" * 50)
        print(f"Total PnL: ${result['total_pnl']:,.2f}")
        print(f"Win Rate: {result['win_rate']:.1%}")
        print(f"Avg Win: ${result['avg_win']:,.2f}")
        print(f"Avg Loss: ${result['avg_loss']:,.2f}")
        print(f"Win/Loss Ratio: {result['win_loss_ratio']:.2f}")
        print(f"Expected Value per Trade: ${result['expected_value']:.2f}")
        print(f"Profit Factor: {result['profit_factor']:.2f}")
        print(f"Max Drawdown: ${abs(result['max_drawdown']):,.2f}")
        print(f"Sharpe Ratio: {result['sharpe']:.2f}")
        
        # Calculate Kelly Criterion
        win_prob = result['win_rate']
        win_loss_ratio = result['win_loss_ratio']
        kelly = win_prob - ((1 - win_prob) / win_loss_ratio)
        
        print("\nPosition Sizing:")
        print(f"Kelly Fraction: {kelly:.1%}")
        print(f"Half Kelly (recommended): {kelly/2:.1%}")
        print(f"Conservative Kelly (1/4): {kelly/4:.1%}")
        
        # Calculate example position sizes
        account_sizes = [10000, 25000, 50000, 100000]
        print("\nExample Position Sizes (using 1/4 Kelly):")
        for size in account_sizes:
            pos_size = size * (kelly/4)
            print(f"${size:,} account: ${pos_size:.2f} per trade")
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot cumulative PnL
    plt.subplot(2, 1, 1)
    for result in results:
        plt.plot(result['cumulative_pnl'], 
                label=f"{result['strategy'].replace('_', ' ').title()}")
    
    plt.title('Cumulative PnL Comparison')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative PnL ($)')
    plt.legend()
    plt.grid(True)
    
    # Plot underwater chart (drawdowns)
    plt.subplot(2, 1, 2)
    for result in results:
        running_max = np.maximum.accumulate(result['cumulative_pnl'])
        drawdown = (result['cumulative_pnl'] - running_max) / running_max
        plt.plot(drawdown,
                label=f"{result['strategy'].replace('_', ' ').title()}")
    
    plt.title('Drawdown Comparison')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown %')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.tight_layout()
    plt.savefig('results/options_strategies_comparison.png')
    plt.close()
    
    # Save detailed results
    summary = pd.DataFrame([{
        'Strategy': r['strategy'].replace('_', ' ').title(),
        'Total PnL': r['total_pnl'],
        'Win Rate': r['win_rate'],
        'Avg Win': r['avg_win'],
        'Avg Loss': r['avg_loss'],
        'Win/Loss Ratio': r['win_loss_ratio'],
        'Expected Value': r['expected_value'],
        'Profit Factor': r['profit_factor'],
        'Max Drawdown': r['max_drawdown'],
        'Sharpe': r['sharpe']
    } for r in results])
    
    summary.to_csv('results/options_strategies_summary.csv', index=False)
    print("\nDetailed results saved to results/options_strategies_summary.csv")

if __name__ == "__main__":
    analyze_options_strategies()
