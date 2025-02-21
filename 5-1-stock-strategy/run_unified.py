import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from download_data import SYMBOLS, load_cached_data
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from config import config as CONFIG
from bear_config import bear_config as BEAR_CONFIG
from itertools import product

# === NEW IMPORT: from your unified strategy file ===
# If your file is named "unified.py" and the function is "run_unified_strategy",
# then do:
from unified import run_unified_strategy

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
    # Add extra days for indicator calculation (30 days for regime window + trading days)
    indicator_days = trading_days + 30  # Add 30 days for proper indicator calculation
    
    shortened_data = {}
    for symbol, df in data.items():
        df = df.sort_index()  # Ensure chronological order
        
        # Get data for indicator calculation first
        last_date = df.index[-1].date()
        start_date = last_date - timedelta(days=indicator_days)
        df = df[df.index.date >= start_date]
        
        # Filter market hours after getting enough data for indicators
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
    # Make a deep copy to avoid modifying original
    import copy
    config = copy.deepcopy(config)
    
    # Extract regime parameters and map them to top-level config
    if 'regime_params' in config:
        for regime, regime_config in config['regime_params'].items():
            # Make a copy of the regime config to avoid modifying original
            regime_config = regime_config.copy()
            
            # Map regime parameters to top-level config with proper prefixes
            if regime == 'bear_high_vol':
                if 'exit_mfi' in regime_config:
                    config['bear_exit'] = regime_config['exit_mfi']
                if 'stop_mult' in regime_config:
                    config[f'{regime}_stop_mult'] = regime_config['stop_mult']
                if 'reward_risk' in regime_config:
                    config[f'{regime}_reward_risk'] = regime_config['reward_risk']
                if 'position_scale' in regime_config:
                    config['bear_position_scale'] = regime_config['position_scale']
                if 'trailing_stop' in regime_config:
                    config['bear_trailing_stop'] = regime_config['trailing_stop']
            elif regime == 'bull_high_vol':
                if 'exit_mfi' in regime_config:
                    config['bull_exit'] = regime_config['exit_mfi']
                if 'stop_mult' in regime_config:
                    config[f'{regime}_stop_mult'] = regime_config['stop_mult']
                if 'reward_risk' in regime_config:
                    config[f'{regime}_reward_risk'] = regime_config['reward_risk']
                if 'position_scale' in regime_config:
                    config['bull_position_scale'] = regime_config['position_scale']
                if 'trailing_stop' in regime_config:
                    config['bull_trailing_stop'] = regime_config['trailing_stop']
    
    # Find parameters that are arrays
    array_params = {k: v for k, v in config.items() if isinstance(v, list)}
    
    # Get base config without arrays
    base_config = {}
    for k, v in config.items():
        if isinstance(v, list):
            base_config[k] = v[0]
        elif isinstance(v, dict):
            base_config[k] = v.copy()
        else:
            base_config[k] = v
    
    # Generate combinations
    from itertools import product
    param_names = list(array_params.keys())
    param_values = [array_params[name] for name in param_names]
    
    combinations = []
    for values in product(*param_values):
        params = base_config.copy()
        for name, value in zip(param_names, values):
            params[name] = value
        combinations.append(params)
    
    # Skip printing parameter combinations
    
    return combinations

def test_strategy(symbol_data, test_config):
    """Test strategy on given data using the unified script"""
    try:
        # Previously: trades = run_portfolio_strategy(symbol_data, test_config)
        # NOW: use the unified strategy approach
        trades = run_unified_strategy(symbol_data, test_config)
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
    
    # Get initial capital from config or fallback
    initial_capital = CONFIG.get('Initial Capital', 100000)
    
    # Get trades with realized PnL (non-zero)
    pnl_trades = [t for t in trades if t['pnl'] != 0]  # Both BUY and SELL exits
    pnls = [t['pnl'] for t in pnl_trades]
    winning_trades = [pnl for pnl in pnls if pnl > 0]
    losing_trades = [pnl for pnl in pnls if pnl <= 0]
    
    # Calculate cumulative portfolio value from trades with PnL
    df['cumulative_pnl'] = 0.0
    for i, row in df.iterrows():
        if row['pnl'] != 0:  # Include both BUY and SELL exits
            df.loc[i, 'cumulative_pnl'] = row['pnl']
    df['cumulative_pnl'] = df['cumulative_pnl'].cumsum()
    df['portfolio_value'] = initial_capital + df['cumulative_pnl']
    
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
    # If 'positions' was recorded in each trade, we can compute:
    max_positions = df['positions'].max() if 'positions' in df.columns else 0
    avg_positions = df['positions'].mean() if 'positions' in df.columns else 0
    
    # Compute minute returns for Sharpe
    df.set_index('datetime', inplace=True)
    df_grouped = df.groupby(df.index)['portfolio_value'].last()
    
    min_time = df_grouped.index.min()
    max_time = df_grouped.index.max()
    regular_index = pd.date_range(min_time, max_time, freq='1min')
    
    portfolio_series = df_grouped.reindex(regular_index).ffill()
    portfolio_series.iloc[0] = portfolio_series.iloc[0] if not pd.isna(portfolio_series.iloc[0]) else initial_capital
    portfolio_series = portfolio_series.ffill()
    
    minute_returns = portfolio_series.pct_change().dropna()
    if len(minute_returns) > 1 and minute_returns.std() != 0:
        # Approx scale by sqrt(# of trading minutes in a year)
        trading_minutes = len(minute_returns)
        # You can approximate 252 trading days * 390 minutes/day = ~ 98280 minutes/year
        # so sqrt(98280) ~ 314
        # but up to you to define
        annual_factor = np.sqrt(252 * 390)
        sharpe = (minute_returns.mean() / minute_returns.std()) * annual_factor
    else:
        sharpe = 0
    
    # Basic metrics
    metrics = {
        'total_pnl': float(np.sum(pnls).item()),
        'n_trades': int(len(pnls)),
        'win_rate': float(len(winning_trades) / len(pnls) if pnls else 0),
        'profit_factor': float(abs(np.sum(winning_trades).item() / np.sum(losing_trades).item()) if losing_trades else float('inf')),
        'avg_win': float(np.mean(winning_trades).item() if winning_trades else 0),
        'avg_loss': float(np.mean(losing_trades).item() if losing_trades else 0),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'max_positions': int(max_positions),
        'avg_positions': float(avg_positions),
        'portfolio_return': float(((final_value - initial_capital) / initial_capital).item()),
        'max_portfolio_value': float(max_value.item()),
        'min_portfolio_value': float(min_value.item()),
        'final_portfolio_value': float(final_value.item())
    }
    return metrics

def print_trades(trades):
    """Print individual trade details with execution information"""
    if not trades:
        print("\nNo trades to display")
        return
        
    print("\nDetailed Trade Analysis:")
    print("-" * 120)
    
    # Sort trades by timestamp
    sorted_trades = sorted(trades, key=lambda x: pd.to_datetime(x.get('timestamp', '1970-01-01')))
    
    for t in sorted_trades:
        ts = pd.to_datetime(t.get('timestamp')).strftime('%Y-%m-%d %H:%M:%S')
        action = t.get('action', '')
        price = t.get('price', 0)
        size = t.get('size', 0)
        pnl = t.get('pnl', 0)
        
        symbol = t.get('symbol', 'Unknown')
        print(f"\nTrade at {ts} - Symbol: {symbol}")
        print(f"Action: {action}")
        print(f"Price: ${price:.2f}")
        print(f"Size: {size}")
        print(f"Regime: {t.get('regime', 'Unknown')}")
        
        # Additional fields
        if 'fill_type' in t:
            print(f"Fill Type: {t['fill_type']}")
            if t['fill_type'] == 'limit':
                print(f"Limit Buffer: ${t['limit_buffer_used']:.3f}")
                print(f"Trigger Price: ${t['trigger_price']:.2f}")
            else:  # market
                print(f"Slippage: ${t.get('slippage',0):.3f}")
                if 'original_price' in t:
                    print(f"Original Price: ${t['original_price']:.2f}")
        
        bar_high = t.get('bar_high', 0)
        bar_low  = t.get('bar_low', 0)
        bar_close= t.get('bar_close', 0)
        print(f"Bar Prices - High: ${bar_high:.2f}, "
              f"Low: ${bar_low:.2f}, "
              f"Close: ${bar_close:.2f}")
        
        if 'reason' in t:
            print(f"Exit Reason: {t['reason']}")
            if 'trigger_type' in t:
                print(f"Trigger Type: {t['trigger_type']}")
        
        if pnl != 0:
            print(f"PnL: ${pnl:.2f}")
        
        print("-" * 60)

def print_metrics(metrics, trades=None, title="Portfolio Performance", score=None):
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
    
    # Print trade details
    # if trades and len(trades) > 0:
    #     print_trades(trades)

def process_param_combination(args):
    """Process a single parameter combination"""
    symbol_data, params = args
    trades, error = test_strategy(symbol_data, params)
    if error:
        return None
        
    metrics = calculate_portfolio_metrics(trades)
    
    # Calculate optimization score
    if metrics['n_trades'] > 0:
        score = (
            metrics['portfolio_return'] * 100 * 0.3 +
            metrics['win_rate'] * 100 * 0.2 +
            min(metrics['profit_factor'], 3) * 10 * 0.2 +
            (1 - metrics['max_drawdown']) * 100 * 0.2 +
            metrics['sharpe'] * 10 * 0.1
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
    
    # Skip printing optimization status
    
    # Use multiprocessing for parallel testing
    n_cores = cpu_count()
    pool = Pool(n_cores)
    
    # Prepare arguments
    args = [(symbol_data, params) for params in combinations]
    
    # Run combinations in parallel with progress bar
    all_results = []
    for result in tqdm(pool.imap_unordered(process_param_combination, args), total=len(args)):
        if result is not None:
            all_results.append(result)
            
    pool.close()
    pool.join()
    
    if not all_results:
        print("No valid results found")
        return config

    # Sort by combined score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    best_result = all_results[0]  # best combination
    
    # Skip printing all results
    
    print("\n" + "="*80)
    print("BEST COMBINATION FOUND")
    print("="*80)
    print(f"Score: {best_result['score']:.2f}")
    print("\nParameters:")
    # Print non-regime parameters
    for k, v in best_result['params'].items():
        if not k.startswith(('bear_high_vol_', 'bull_high_vol_')) and k != 'regime_params':
            print(f"{k}: {v}")
    # Print bear regime parameters
    print("\nBear High Vol Parameters:")
    for k, v in best_result['params'].items():
        if k.startswith('bear_high_vol_'):
            print(f"  {k.replace('bear_high_vol_', '')}: {v}")
    # Print bull regime parameters
    print("\nBull High Vol Parameters:")
    for k, v in best_result['params'].items():
        if k.startswith('bull_high_vol_'):
            print(f"  {k.replace('bull_high_vol_', '')}: {v}")
    print("\nPerformance:")
    print_metrics(best_result['metrics'], None, "Best Strategy Performance", float(best_result['score']))
    print("\nDetailed Trade Analysis:")
    # print_trades(best_result['trades'])
    
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
    
    # Run optimization for both configs
    print("\nOptimizing Bull Strategy...")
    bull_params = optimize_strategy(symbol_data, CONFIG)
    
    print("\nOptimizing Bear Strategy...")
    bear_params = optimize_strategy(symbol_data, BEAR_CONFIG)

if __name__ == '__main__':
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    main(trading_days=days)
