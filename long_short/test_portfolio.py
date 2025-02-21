import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from long_short.main import run_backtest

def test_portfolio_construction():
    """
    Test the portfolio construction and backtesting functionality using real market data.
    """
    # Expanded set of instruments including different sectors
    symbols = [
        # Tech stocks
        'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD',
        # ETFs for broader market exposure
        'SPY', 'QQQ', 'TLT', 'SQQQ',
    ]
    
    # Parameters for more thorough analysis
    window_size = 120  # Increased window size for better pattern recognition
    step_size = 1     # Keep step size at 1 for maximum data utilization
    
    print("\nStarting comprehensive portfolio backtest...")
    print(f"Using data from: momentum_ai_trading/data/processed/")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Window size: {window_size}")
    print(f"Step size: {step_size}")
    print("\n" + "="*50)
    
    try:
        # Run backtest
        results = run_backtest(symbols, window_size, step_size)
        
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'momentum_ai_trading/results/deep_backtest/run_{timestamp}'
        os.makedirs(results_path, exist_ok=True)
        
        # Save detailed metrics
        metrics_df = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annualized Sharpe Ratio',
                'Maximum Drawdown',
                'Number of Instruments',
                'Window Size',
                'Step Size'
            ],
            'Value': [
                f"{results['total_return']:.2%}",
                f"{results['sharpe_ratio']:.2f}",
                f"{results['max_drawdown']:.2%}",
                len(symbols),
                window_size,
                step_size
            ]
        })
        
        metrics_df.to_csv(f'{results_path}/metrics.csv', index=False)
        
        # Save symbols list
        with open(f'{results_path}/symbols.txt', 'w') as f:
            f.write('\n'.join(symbols))
        
        # Create summary report
        with open(f'{results_path}/summary.txt', 'w') as f:
            f.write("Deep Learning Long-Short Portfolio Backtest Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Run Date: {timestamp}\n\n")
            f.write("Portfolio Composition:\n")
            f.write(f"Number of instruments: {len(symbols)}\n")
            f.write(f"Instruments: {', '.join(symbols)}\n\n")
            f.write("Model Parameters:\n")
            f.write(f"Window size: {window_size}\n")
            f.write(f"Step size: {step_size}\n\n")
            f.write("Performance Metrics:\n")
            f.write(f"Total Return: {results['total_return']:.2%}\n")
            f.write(f"Annualized Sharpe Ratio: {results['sharpe_ratio']:.2f}\n")
            f.write(f"Maximum Drawdown: {results['max_drawdown']:.2%}\n")
        
        print("\nBacktest completed successfully!")
        print(f"Results saved to: {results_path}")
        print("\nKey Metrics:")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_portfolio_construction()
