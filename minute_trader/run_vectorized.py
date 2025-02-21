import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from download_data import SYMBOLS, load_cached_data
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from config import config as CONFIG
from portfolio_strategy import run_portfolio_strategy
from itertools import product

def load_data(symbols):
    """Load data from cache for all symbols with progress bar"""
    data = {}
    print("\nLoading cached data...")
    for symbol in tqdm(symbols, desc="Loading symbols"):
        df = load_cached_data(symbol)
        if df is not None and not df.empty:
            data[symbol] = df
        else:
            print(f"\nNo cached data found for {symbol}")
    return data

def prepare_data(data, trading_days=3):
    """
    Prepare data for strategy testing.
    Args:
        data: Dictionary of symbol -> DataFrame
        trading_days: Number of days to test (default 3)
    """
    # Add extra days for indicator calculation
    indicator_days = trading_days + 2  # Add 2 extra days for indicators
    
    shortened_data = {}
    for symbol, df in data.items():
        # Get last N trading days during market hours (9:30 AM - 4:00 PM ET)
        df = df.sort_index()  # Ensure chronological order
        
        # Get data for indicator calculation
        last_date = df.index[-1].date()
        start_date = last_date - timedelta(days=indicator_days)
        df = df[df.index.date >= start_date]
        
        # Get market hours data for the last 3 days
        market_hours = df[
            (df.index.time >= datetime.strptime('09:30', '%H:%M').time()) &
            (df.index.time <= datetime.strptime('16:00', '%H:%M').time())
        ]
        
        # Group by date and get last N days
        dates = market_hours.groupby(market_hours.index.date).size().index
        if len(dates) >= trading_days:
            last_n_dates = dates[-trading_days:]
            print(f"\n{symbol} trading dates: {', '.join(str(d) for d in last_n_dates)}")
            # Keep all data but mark which dates are trading days
            date_mask = pd.to_datetime(df.index.date).isin(pd.to_datetime(last_n_dates))
            df_short = df[date_mask]
        else:
            print(f"Warning: {symbol} has less than {trading_days} days of data")
            df_short = df[
                (df.index.time >= datetime.strptime('09:30', '%H:%M').time()) &
                (df.index.time <= datetime.strptime('16:00', '%H:%M').time())
            ]
        shortened_data[symbol] = df_short
        print(f"{symbol}: {len(df_short)} bars ({df_short.index[0]} -> {df_short.index[-1]})")
    return shortened_data

def generate_param_combinations(config):
    """Generate all parameter combinations from arrays in config"""
    # Find parameters that are arrays
    array_params = {k: v for k, v in config.items() if isinstance(v, list)}
    
    # Get base config without arrays
    base_config = {}
    for k, v in config.items():
        if isinstance(v, list):
            # For list parameters, use first value as default in base config
            base_config[k] = v[0]
        elif isinstance(v, dict):
            base_config[k] = v.copy()
        else:
            base_config[k] = v
    
    # Generate combinations
    param_names = list(array_params.keys())
    param_values = [array_params[name] for name in param_names]
    
    combinations = []
    for values in product(*param_values):
        params = base_config.copy()
        for name, value in zip(param_names, values):
            params[name] = value
        combinations.append(params)
    
    print("\nTesting combinations with parameters:")
    for params in combinations:
        print("\nParameters:")
        for k, v in params.items():
            if k in array_params:
                print(f"{k}: {v}")
    
    return combinations

def test_strategy(symbol_data, test_config):
    """Test strategy on given data"""
    try:
        trades = run_portfolio_strategy(symbol_data, test_config)
        return trades, None
    except Exception as e:
        return [], f"Error testing strategy: {str(e)}"

def calculate_portfolio_metrics(trades):
    """Calculate portfolio-level performance metrics"""
    if not trades:
        return {
            'total_pnl': 0,
            'n_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'max_positions': 0,
            'avg_positions': 0,
            'portfolio_return': 0,
            'max_portfolio_value': 0,
            'min_portfolio_value': float('inf'),
            'final_portfolio_value': 0
        }

    # Convert trades to DataFrame for analysis
    df = pd.DataFrame(trades)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('datetime')
    
    # Get initial capital from first trade
    initial_capital = CONFIG.get('Initial Capital', 100000)
    
    # Calculate trade metrics
    sell_trades = [t for t in trades if t['action'] == 'SELL']
    pnls = [t['pnl'] for t in sell_trades]
    winning_trades = [pnl for pnl in pnls if pnl > 0]
    losing_trades = [pnl for pnl in pnls if pnl <= 0]
    
    # Calculate portfolio value at each trade
    df['cumulative_pnl'] = 0.0
    for i, row in df.iterrows():
        if row['action'] == 'SELL':
            df.loc[i, 'cumulative_pnl'] = row['pnl']
    df['cumulative_pnl'] = df['cumulative_pnl'].cumsum()
    df['portfolio_value'] = initial_capital + df['cumulative_pnl']
    
    # Portfolio value metrics
    portfolio_values = df['portfolio_value'].values
    final_value = portfolio_values[-1]
    max_value = np.max(portfolio_values)
    min_value = np.min(portfolio_values)
    
    # Calculate drawdown
    peak = initial_capital
    max_drawdown = 0
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    # Position metrics
    max_positions = df['positions'].max()
    avg_positions = df['positions'].mean()
    
    # Calculate minute returns for Sharpe ratio using regular intervals
    # First group trades by timestamp to handle duplicates
    df.set_index('datetime', inplace=True)
    df_grouped = df.groupby(df.index)['portfolio_value'].last()
    
    # Create regular time series
    min_time = df_grouped.index.min()
    max_time = df_grouped.index.max()
    regular_index = pd.date_range(min_time, max_time, freq='1min')
    
    # Forward fill portfolio values to get value at each minute
    portfolio_series = df_grouped.reindex(regular_index).ffill()
    if portfolio_series.iloc[0] is pd.NA:  # Fill initial value if needed
        portfolio_series.iloc[0] = initial_capital
    portfolio_series = portfolio_series.ffill()
    
    # Calculate returns
    minute_returns = portfolio_series.pct_change().dropna()
    
    # Annualize using actual trading minutes
    # Annualize using actual trading minutes
    trading_minutes = len(minute_returns)
    annualization = np.sqrt(250 * trading_minutes)
    
    metrics = {
        'total_pnl': float(sum(pnls)),
        'n_trades': int(len(pnls)),
        'win_rate': float(len(winning_trades) / len(pnls) if pnls else 0),
        'profit_factor': float(abs(sum(winning_trades) / sum(losing_trades)) if losing_trades else float('inf')),
        'avg_win': float(np.mean(winning_trades) if winning_trades else 0),
        'avg_loss': float(np.mean(losing_trades) if losing_trades else 0),
        'sharpe': float(annualization * (minute_returns.mean() / minute_returns.std()) / days if len(minute_returns) > 1 else 0),
        'max_drawdown': float(max_drawdown),
        'max_positions': int(max_positions),
        'avg_positions': float(avg_positions),
        'portfolio_return': float(np.array((final_value - initial_capital) / initial_capital).item()),
        'max_portfolio_value': float(max_value),
        'min_portfolio_value': float(min_value),
        'final_portfolio_value': float(final_value)
    }
    return metrics
def print_trades(trades):
    """Print individual trade details"""
    if not trades:
        print("\nNo trades to display")
        return
        
    print("\nIndividual Trades:")
    print(f"{'Timestamp':<25} {'Symbol':<6} {'Action':<6} {'Price':>10} {'Size':>6} {'PnL':>12} {'Reason':<15}")
    print("-" * 85)
    
    # Debug print
    print("\nDebug: First trade structure:")
    print(trades[0])
    
    # Sort trades by timestamp
    sorted_trades = sorted(trades, key=lambda x: pd.to_datetime(x.get('timestamp', '1970-01-01')))
    
    for t in sorted_trades:
        # Get timestamp with default
        ts = t.get('timestamp')
        if isinstance(ts, pd.Timestamp):
            timestamp = ts.strftime('%Y-%m-%d %H:%M:%S')
        else:
            try:
                timestamp = pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')
            except:
                timestamp = ''
        
        # Coerce numeric fields to safe values
        pnl = t.get('pnl')
        pnl_str = f"${pnl:,.2f}" if pnl is not None else ''
        
        # Reason could be None
        reason_value = t.get('reason', '')
        reason_str = str(reason_value) if reason_value is not None else ''
        
        # Get values with defaults for None
        symbol = t.get('symbol', '')
        action = t.get('action', '')
        price = t.get('price', 0)
        size = t.get('size', 0)
        
        # Format as needed
        ts_str = str(timestamp) if timestamp else ''
        sym_str = str(symbol) if symbol else ''
        act_str = str(action) if action else ''
        price_str = f"${price:,.2f}" if price is not None else "$0.00"
        size_str = str(size) if size is not None else '0'
        
        print(f"{ts_str:<25} {sym_str:<6} {act_str:<6} "
              f"{price_str:>10} {size_str:>6} {pnl_str:>12} {reason_str:<15}")


def print_metrics(metrics, trades=None, title="Portfolio Performance", score=None, print_trades_detail=False):
    """Print portfolio metrics in a formatted way"""
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")
    
    # Print overall metrics
    print("Portfolio Performance:")
    if score is not None:
        print(f"Optimization Score: {float(score):.2f}")
    print(f"Net PnL: ${metrics['total_pnl']:,.2f}")
    print(f"Portfolio Return: {metrics['portfolio_return']:.1%}")
    print(f"Final Portfolio Value: ${metrics['final_portfolio_value']:,.2f}")
    print(f"Max Portfolio Value: ${metrics['max_portfolio_value']:,.2f}")
    print(f"Min Portfolio Value: ${metrics['min_portfolio_value']:,.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
    print(f"\nTrading Statistics:")
    print(f"Total Trades: {metrics['n_trades']}")
    if metrics['n_trades'] > 0:
        print(f"Avg PnL per Trade: ${metrics['total_pnl']/metrics['n_trades']:,.2f}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Avg Win: ${metrics['avg_win']:,.2f}")
    print(f"Avg Loss: ${metrics['avg_loss']:,.2f}")
    print(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"\nPosition Management:")
    print(f"Max Positions: {metrics['max_positions']}")
    print(f"Avg Positions: {metrics['avg_positions']:.1f}")
    
    if print_trades_detail and trades and len(trades) > 0:
        print_trades(trades)

def process_param_combination(args):
    """Process a single parameter combination"""
    symbol_data, params = args
    trades, error = test_strategy(symbol_data, params)
    if error:
        return None
        
    metrics = calculate_portfolio_metrics(trades)
    
    # Calculate optimization score
    if metrics['n_trades'] > 0:
        # Combine metrics with emphasis on consistency
        score = (
            metrics['portfolio_return'] * 100 * 0.3 +     # 30% weight on total return
            metrics['win_rate'] * 100 * 0.2 +            # 20% weight on win rate
            min(metrics['profit_factor'], 3) * 10 * 0.2 + # 20% weight on profit factor
            (1 - metrics['max_drawdown']) * 100 * 0.2 +  # 20% weight on drawdown control
            metrics['sharpe'] * 10 * 0.1                 # 10% weight on Sharpe ratio
        )
    else:
        score = float('-inf')
        
    return {
        'params': params,
        'metrics': metrics,
        'score': score,
        'trades': trades
    }

def optimize_strategy(symbol_data, config):
    """Test different parameter combinations to find optimal settings"""
    combinations = generate_param_combinations(config)
    
    print("\nOptimizing strategy parameters...")
    print(f"Testing {len(combinations)} combinations")
    
    # Create pool of workers
    n_cores = cpu_count()
    pool = Pool(n_cores)
    
    # Prepare arguments for parallel processing
    args = [(symbol_data, params) for params in combinations]
    
    # Run combinations in parallel with progress bar
    all_results = []
    for result in tqdm(pool.imap_unordered(process_param_combination, args), total=len(args)):
        if result is not None:
            all_results.append(result)
            
    # Close pool
    pool.close()
    pool.join()
    
    if not all_results:
        print("No valid results found")
        return config

    # Sort by combined score and print all results
    all_results.sort(key=lambda x: x['score'], reverse=True)
    best_result = all_results[0]  # Get best result

    print("\nAll combinations (sorted by combined score):")
    for r in all_results:
        print("\nParameters:")
        for param, value in r['params'].items():
            if isinstance(value, dict):
                continue  # Skip printing complex params for clarity
            print(f"{param}: {value}")
        print(f"Score: {float(r['score']):.2f}")
        print_metrics(r['metrics'], r['trades'], "Combination Performance", float(r['score']))
    
    print("\nBest Parameters Found:")
    print("=" * 50)
    for param, value in best_result['params'].items():
        if isinstance(value, dict):
            continue  # Skip printing complex params for clarity
        print(f"{param}: {value}")
    print("=" * 50)
    
    print("\nBest Parameters Performance:")
    print_metrics(best_result['metrics'], best_result['trades'], "Best Parameters Performance", float(best_result['score']))
    
    return best_result['params']

def main(trading_days=3):
    """
    Run strategy optimization
    Args:
        trading_days: Number of days to test (default 3)
    """
    # Load cached data
    print("Loading cached data...")
    data = load_data(SYMBOLS)
    
    if not data:
        print("No cached data found. Please run download_data.py first.")
        return
        
    # Prepare data
    print(f"Preparing data for analysis (last {trading_days} days)...")
    symbol_data = prepare_data(data, trading_days=trading_days)
    
    # Run optimization
    best_params = optimize_strategy(symbol_data, CONFIG)

if __name__ == '__main__':
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(trading_days=days)
