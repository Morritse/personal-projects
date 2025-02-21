import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def simulate_realistic_options(win_rate=0.60, num_trades=1000):
    """Simulate options trading with realistic constraints."""
    np.random.seed(42)
    
    # More realistic parameters
    risk_per_trade = 100  # Still risk $100 per trade
    
    # Account for real-world factors
    entry_slippage = 0.10  # Lose 10% to bid-ask spread on entry
    exit_slippage = 0.10   # Lose 10% to bid-ask spread on exit
    theta_decay = 0.05     # Lose 5% per day to time decay
    
    # Generate trades
    trades = []
    for i in range(num_trades):
        # Determine if direction is correct
        is_direction_right = np.random.random() < win_rate
        
        if is_direction_right:
            # Even when direction is right:
            # 1. Small move (1.5x): 40% chance
            # 2. Medium move (2x): 30% chance
            # 3. Large move (3x): 20% chance
            # 4. Very large move (4x): 10% chance
            move_type = np.random.random()
            
            if move_type < 0.40:  # 40% chance
                reward_multiplier = 1.5
            elif move_type < 0.70:  # 30% chance
                reward_multiplier = 2.0
            elif move_type < 0.90:  # 20% chance
                reward_multiplier = 3.0
            else:  # 10% chance
                reward_multiplier = 4.0
                
            # Calculate holding period (1-5 days)
            days_held = np.random.randint(1, 6)
            
            # Calculate PnL
            gross_profit = risk_per_trade * reward_multiplier
            entry_cost = risk_per_trade * entry_slippage
            exit_cost = gross_profit * exit_slippage
            theta_cost = risk_per_trade * theta_decay * days_held
            
            pnl = gross_profit - entry_cost - exit_cost - theta_cost
            
        else:
            # When direction is wrong:
            # 1. Small loss (60%): Lose 60-80% of premium
            # 2. Full loss (40%): Lose 100% of premium
            if np.random.random() < 0.60:
                loss_multiplier = np.random.uniform(0.6, 0.8)
            else:
                loss_multiplier = 1.0
                
            pnl = -risk_per_trade * loss_multiplier
        
        trades.append({
            'pnl': pnl,
            'direction_right': is_direction_right
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(trades)
    
    # Calculate metrics
    total_pnl = df['pnl'].sum()
    win_rate = len(df[df['pnl'] > 0]) / len(df)
    avg_win = df[df['pnl'] > 0]['pnl'].mean()
    avg_loss = df[df['pnl'] < 0]['pnl'].mean()
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    
    # Calculate drawdown
    cumulative = df['pnl'].cumsum()
    running_max = cumulative.expanding().max()
    drawdowns = cumulative - running_max
    max_drawdown = drawdowns.min()
    
    # Calculate daily returns for Sharpe
    trades_per_day = 2  # Assume 2 trades per day (more realistic)
    days = num_trades // trades_per_day
    daily_pnl = np.array_split(df['pnl'].values, days)
    daily_returns = np.array([sum(day) for day in daily_pnl])
    sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) != 0 else 0
    
    # Calculate win metrics by size
    wins = df[df['pnl'] > 0]['pnl']
    print("\nWin Distribution:")
    print(f"Small Wins (< 2x): {len(wins[wins < risk_per_trade * 2])} trades")
    print(f"Medium Wins (2-3x): {len(wins[(wins >= risk_per_trade * 2) & (wins < risk_per_trade * 3)])} trades")
    print(f"Large Wins (> 3x): {len(wins[wins >= risk_per_trade * 3])} trades")
    
    # Calculate expected value
    expected_value = df['pnl'].mean()
    
    # Calculate required win rate for breakeven
    breakeven_wr = abs(avg_loss) / (abs(avg_loss) + avg_win)
    
    print("\nRealistic Options Strategy Results:")
    print("-" * 50)
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Avg Win: ${avg_win:.2f}")
    print(f"Avg Loss: ${avg_loss:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Max Drawdown: ${abs(max_drawdown):,.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Expected Value per Trade: ${expected_value:.2f}")
    print(f"Required Win Rate for Breakeven: {breakeven_wr:.1%}")
    
    # Plot equity curve
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(cumulative)
    plt.title('Equity Curve')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative PnL ($)')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(drawdowns)
    plt.title('Drawdown')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown ($)')
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/realistic_options_results.png')
    plt.close()
    
    # Calculate Kelly Criterion
    win_loss_ratio = abs(avg_win / avg_loss)
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    print("\nPosition Sizing:")
    print(f"Kelly Fraction: {kelly:.1%}")
    print(f"Half Kelly (recommended): {kelly/2:.1%}")
    print(f"Quarter Kelly (conservative): {kelly/4:.1%}")
    
    # Example position sizes
    account_sizes = [10000, 25000, 50000, 100000]
    print("\nExample Position Sizes (using 1/4 Kelly):")
    for size in account_sizes:
        pos_size = size * (kelly/4)
        print(f"${size:,} account: ${pos_size:.2f} per trade")
    
    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe': sharpe,
        'expected_value': expected_value,
        'breakeven_wr': breakeven_wr,
        'kelly': kelly
    }

if __name__ == "__main__":
    print("Simulating 1000 trades with realistic options constraints...")
    print("Including: slippage, theta decay, and varied profit targets")
    simulate_realistic_options()
