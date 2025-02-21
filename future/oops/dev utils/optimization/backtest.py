import pandas as pd
from fetch_futures_data import FUTURES
from strategy import FuturesStrategy, analyze_portfolio
from config import MA_STRATEGY

def load_data():
    """Load data for all futures contracts"""
    dfs = {}
    for symbol in FUTURES.keys():
        try:
            filename = f"data/{symbol.replace('=', '_')}_daily.csv"
            df = pd.read_csv(filename, index_col=0, parse_dates=True)
            dfs[symbol] = df
        except FileNotFoundError:
            print(f"No data file found for {symbol}")
    return dfs

def analyze_instrument(df: pd.DataFrame, strategy: FuturesStrategy, symbol: str) -> None:
    """Analyze and print detailed metrics for a single instrument"""
    df_result, metrics = strategy.run_strategy(df.copy())
    
    # Only consider periods where signals are valid
    valid_mask = df_result['signal_valid']
    valid_returns = df_result.loc[valid_mask, 'strat_returns']
    
    # Calculate additional metrics
    avg_pos_size = abs(df_result.loc[valid_mask, 'position']).mean()
    trade_count = (df_result.loc[valid_mask, 'position'] != 
                  df_result.loc[valid_mask, 'position'].shift(1)).sum()
    
    print(f"\n{FUTURES[symbol]}:")
    print(f"  Annual Return: {metrics['annual_return']:.2%}")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  Win Rate: {metrics['win_rate']:.2%}")
    print(f"  Average Position Size: {avg_pos_size:.2%}")
    print(f"  Number of Trades: {trade_count}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Calculate monthly returns
    if isinstance(df_result.index, pd.DatetimeIndex):
        monthly_returns = valid_returns.resample('M').sum()
        print(f"  Best Month: {monthly_returns.max():.2%}")
        print(f"  Worst Month: {monthly_returns.min():.2%}")

def main():
    # Load data
    print("Loading futures data...")
    dfs = load_data()
    
    if not dfs:
        print("No data files found. Please run fetch_futures_data.py first.")
        return
    
    # Initialize strategy with parameters from config
    strategy = FuturesStrategy(**MA_STRATEGY)
    
    # Analyze individual instruments
    print("\nIndividual Instrument Metrics:")
    print("Note: All metrics calculated only after initial lookback period")
    for symbol, df in dfs.items():
        analyze_instrument(df, strategy, symbol)
    
    # Run portfolio analysis
    print("\nAnalyzing portfolio...")
    portfolio_metrics = analyze_portfolio(dfs, strategy)
    
    # Print portfolio results
    print("\nPortfolio Metrics (Correlation-Weighted):")
    print(f"Annual Return: {portfolio_metrics['annual_return']:.2%}")
    print(f"Annual Volatility: {portfolio_metrics['annual_vol']:.2%}")
    print(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {portfolio_metrics['max_drawdown']:.2%}")

if __name__ == "__main__":
    main()
    print(f"Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {portfolio_metrics['max_drawdown']:.2%}")

if __name__ == "__main__":
    main()
