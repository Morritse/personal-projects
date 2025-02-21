import asyncio
from backtest_runner import BacktestRunner

async def main():
    from dotenv import load_dotenv
    import os
    
    # Load environment variables
    load_dotenv()
    
    # Initialize runner with just SPY for testing
    runner = BacktestRunner(
        symbols=[
            # Major ETFs
            'SPY',   # S&P 500
            'QQQ',   # Nasdaq 100
            'IWM',   # Russell 2000
            
            # Sector ETFs
            'XLK',   # Technology
            'XLF',   # Financials
            'XLE',   # Energy
            
            # Individual Stocks
            'AAPL',  # Large Cap Tech
            'JPM',   # Large Cap Finance
            'XOM'    # Large Cap Energy
        ],
        start_date='2024-07-01',  # 6 months of data
        end_date='2024-12-28',    # Avoid last trading day
        timeframes=['short', 'medium', 'long'],
        initial_capital=100000.0,
        commission=0.001  # 0.1% commission rate
    )
    
    # Fetch historical data
    print("Fetching historical data...")
    await runner.fetch_historical_data()
    
    # First run a backtest with live trading parameters
    print("\nRunning backtest with live trading parameters...")
    live_metrics = runner.run_backtest({
        'trend_weight': 0.3,      # Weight for trend signals (increased for parent's signal combo)
        'momentum_weight': 0.4,    # Weight for momentum signals (increased for stronger signals)
        'reversal_weight': 0.3,    # Weight for reversal signals (increased for parent's signal combo)
        'breakout_threshold': 0.2, # Strong signal threshold (lowered for more trades)
        'strong_threshold': 0.1,   # Medium signal threshold (lowered for more trades)
        'weak_threshold': 0.03     # Weak signal threshold (lowered for more trades)
    })
    
    print("\nLive Trading Results:")
    print(f"Return: {live_metrics['return_pct']:.2%}")
    print(f"Trades: {live_metrics['profitable_trades']}/{live_metrics['total_trades']} ({live_metrics['win_rate']:.1%})")
    
    # Run optimization with parallel processing
    print("\nRunning parallel optimization:")
    print("1. Pre-calculate indicators for all symbols")
    print("2. Split data into 3 time periods")
    print("3. Test 729 parameter combinations in parallel")
    param_ranges = {
        'trend_weight': (0.1, 0.4),       # Wider range for trend
        'momentum_weight': (0.2, 0.5),    # Wider range for momentum
        'reversal_weight': (0.1, 0.4),    # Wider range for reversal
        'breakout_threshold': (0.1, 0.3), # Lower threshold for more trades
        'strong_threshold': (0.05, 0.15), # Lower threshold for more trades
        'weak_threshold': (0.01, 0.05)    # Lower threshold for more trades
    }
    
    # Optimize using walk-forward analysis
    optimization_results = await runner.optimize_parameters(
        param_ranges=param_ranges,
        n_iterations=3,   # Grid search: 3^6 = 729 combinations
        n_splits=3       # Faster validation
    )
    
    # Run backtest with optimized parameters
    print("\nRunning backtest with optimized parameters...")
    optimized_metrics = runner.run_backtest(
        strategy_params=optimization_results['best_params']
    )
    
    print("\nOptimized Results:")
    print(f"Return: {optimized_metrics['return_pct']:.2%}")
    print(f"Trades: {optimized_metrics['profitable_trades']}/{optimized_metrics['total_trades']} ({optimized_metrics['win_rate']:.1%})")
    
    print("\nBest Parameters (from parallel grid search):")
    print("Tested 729 combinations across 3 time periods")
    for param, value in optimization_results['best_params'].items():
        print(f"{param}: {value:.4f}")

if __name__ == '__main__':
    asyncio.run(main())
