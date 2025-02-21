import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import itertools
import datetime

# Optimized MA parameters from previous run
OPTIMIZED_PARAMS = {
    "lookback_fast": 20,
    "lookback_slow": 200,
    "vol_lookback": 60,
    "vol_target": 0.15,
    "adx_threshold": None
}

# Trading execution parameters to optimize
TRADING_GRID = {
    # First partial exit threshold in ATR multiples
    'partial_exit_1': [0.0, 0.75, 1.0, 1.5],
    
    # Second partial exit threshold in ATR multiples
    'partial_exit_2': [0.0, 1.0, 1.5, 2.0],
    
    # Time-based exit in days
    'time_stop': [0, 20, 30, 60],
    
    # Trailing stop factor in ATR multiples
    'trailing_stop_factor': [0.0, 1.5, 2.5, 3.0],
    
    # Position scaling approach
    'scaling_mode': ['none', 'pyramid'],
}

def load_data(data_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Loads CSV files from data_folder.
    """
    data_dict = {}
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    for path in csv_files:
        filename = os.path.basename(path)
        symbol = filename.replace(".csv", "")
        
        try:
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df.sort_values("Date", inplace=True)
            df.set_index("Date", inplace=True)
            
            df = df[['open','high','low','close','volume']].copy()
            df.dropna(subset=['open','high','low','close'], inplace=True)
            
            for col in ['open','high','low','close','volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(inplace=True)

            print(f"\nLoaded {symbol}:")
            print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"  Rows: {len(df)}")
            print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            data_dict[symbol] = df
            
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)}")
            continue
    
    return data_dict

def compute_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute technical indicators using optimized MA parameters.
    """
    df = df.copy()
    
    # Use optimized parameters
    lookback_fast = OPTIMIZED_PARAMS['lookback_fast']
    lookback_slow = OPTIMIZED_PARAMS['lookback_slow']
    vol_lookback = OPTIMIZED_PARAMS['vol_lookback']
    
    # Returns and volatility
    returns = df['close'].pct_change()
    df['returns'] = returns.replace([np.inf, -np.inf], np.nan)
    
    # Moving averages
    df['ma_fast'] = df['close'].rolling(window=lookback_fast, min_periods=1).mean()
    df['ma_slow'] = df['close'].rolling(window=lookback_slow, min_periods=1).mean()
    
    # ATR calculation
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=vol_lookback, min_periods=1).mean()

    # Volatility
    df['roll_std'] = df['returns'].rolling(window=vol_lookback, min_periods=1).std() * np.sqrt(252)
    df['roll_std'] = df['roll_std'].replace(0, np.nan)
    
    return df

def apply_exits(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply various exit rules based on trading parameters.
    """
    df = df.copy()
    
    # Initialize tracking columns with proper dtypes
    df['days_in_trade'] = 0
    df['exit_type'] = None
    df['exit_level_1'] = None
    df['exit_level_2'] = None
    df['trailing_stop'] = None
    df['position'] = pd.Series(0, index=df.index, dtype=np.float64)  # Initialize as float64
    
    position = 0
    entry_price = None
    days_held = 0
    position_size = 1.0  # Start with full size
    
    for i in range(1, len(df)):
        curr_price = df.iloc[i]['close']
        curr_atr = df.iloc[i]['atr']
        
        # Update position tracking
        if position != 0:
            days_held += 1
            df.iloc[i, df.columns.get_loc('days_in_trade')] = days_held
            
            # Time stop
            if params['time_stop'] > 0 and days_held >= params['time_stop']:
                position = 0
                entry_price = None
                days_held = 0
                position_size = 1.0
                df.iloc[i, df.columns.get_loc('exit_type')] = 'time'
                continue
            
            # Partial exits
            if position_size == 1.0 and params['partial_exit_1'] > 0:
                if (position > 0 and curr_price >= entry_price + params['partial_exit_1'] * curr_atr) or \
                   (position < 0 and curr_price <= entry_price - params['partial_exit_1'] * curr_atr):
                    position_size = 0.5
                    df.iloc[i, df.columns.get_loc('exit_type')] = 'partial1'
            
            if position_size == 0.5 and params['partial_exit_2'] > 0:
                if (position > 0 and curr_price >= entry_price + params['partial_exit_2'] * curr_atr) or \
                   (position < 0 and curr_price <= entry_price - params['partial_exit_2'] * curr_atr):
                    position_size = 0.25
                    df.iloc[i, df.columns.get_loc('exit_type')] = 'partial2'
            
            # Trailing stop
            if params['trailing_stop_factor'] > 0:
                if position > 0:
                    trail_level = curr_price - params['trailing_stop_factor'] * curr_atr
                    if df.iloc[i-1]['trailing_stop'] is not None:
                        trail_level = max(trail_level, df.iloc[i-1]['trailing_stop'])
                    df.iloc[i, df.columns.get_loc('trailing_stop')] = trail_level
                    
                    if curr_price < trail_level:
                        position = 0
                        entry_price = None
                        days_held = 0
                        position_size = 1.0
                        df.iloc[i, df.columns.get_loc('exit_type')] = 'trail'
                else:
                    trail_level = curr_price + params['trailing_stop_factor'] * curr_atr
                    if df.iloc[i-1]['trailing_stop'] is not None:
                        trail_level = min(trail_level, df.iloc[i-1]['trailing_stop'])
                    df.iloc[i, df.columns.get_loc('trailing_stop')] = trail_level
                    
                    if curr_price > trail_level:
                        position = 0
                        entry_price = None
                        days_held = 0
                        position_size = 1.0
                        df.iloc[i, df.columns.get_loc('exit_type')] = 'trail'
        
        # Entry signals (using optimized MA parameters)
        if position == 0:
            if df.iloc[i]['ma_fast'] > df.iloc[i]['ma_slow']:
                position = 1
                entry_price = curr_price
                days_held = 0
                position_size = 1.0
            elif df.iloc[i]['ma_fast'] < df.iloc[i]['ma_slow']:
                position = -1
                entry_price = curr_price
                days_held = 0
                position_size = 1.0
        
        # Store position
        df.iloc[i, df.columns.get_loc('position')] = position * position_size
    
    return df

def backtest_symbol(df: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run backtest with trading parameters.
    """
    df = compute_indicators(df, params)
    df = apply_exits(df, params)
    
    # Shift position for next-day execution
    df['position_shifted'] = df['position'].shift(1).fillna(0)
    
    # Calculate returns
    df['strategy_ret'] = df['position_shifted'] * df['returns']
    
    return df['strategy_ret'], df

def run_backtest_on_universe(
    data_dict: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Run backtest across all symbols.
    """
    all_rets = pd.DataFrame()
    
    for symbol, df_raw in data_dict.items():
        daily_strat, _ = backtest_symbol(df_raw.copy(), params)
        daily_strat.name = symbol
        all_rets = pd.concat([all_rets, daily_strat], axis=1)
    
    all_rets.dropna(how='all', inplace=True)
    all_rets['portfolio'] = all_rets.mean(axis=1)
    return all_rets

def compute_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Compute performance metrics.
    """
    ann_factor = 252
    
    rets = returns.dropna()
    if len(rets) < 2:
        return {
            "annual_return": 0,
            "annual_vol": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0
        }
    
    mean_daily = rets.mean()
    annual_ret = mean_daily * ann_factor
    
    daily_std = rets.std()
    annual_vol = daily_std * np.sqrt(ann_factor)
    
    sharpe = annual_ret / annual_vol if annual_vol > 1e-9 else 0
    
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1).min()
    
    return {
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd
    }

def param_combinations(param_grid: Dict[str, List[Any]]):
    """
    Generate parameter combinations.
    """
    keys = list(param_grid.keys())
    for combo_vals in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, combo_vals))

def main_optimization(data_dict: Dict[str, pd.DataFrame], max_runs=1000):
    """
    Run optimization of trading parameters.
    """
    best_sharpe = -999
    best_params = None
    best_metrics = {}
    best_returns = None
    
    run_count = 0
    for param_set in param_combinations(TRADING_GRID):
        run_count += 1
        if run_count > max_runs:
            break
            
        all_rets = run_backtest_on_universe(data_dict, param_set)
        portfolio_rets = all_rets['portfolio']
        metrics = compute_metrics(portfolio_rets)
        
        print(f"\nRun {run_count}:")
        print(f"  Params: exits=({param_set['partial_exit_1']}, {param_set['partial_exit_2']}), "
              f"time={param_set['time_stop']}, trail={param_set['trailing_stop_factor']}, "
              f"scale={param_set['scaling_mode']}")
        print(f"  Results: Sharpe: {metrics['sharpe_ratio']:.2f}, Return: {metrics['annual_return']:.1%}, "
              f"MaxDD: {metrics['max_drawdown']:.1%}")
        
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_params = param_set
            best_metrics = metrics
            best_returns = all_rets
    
    print("\n=== Best Result ===")
    print(f"Sharpe: {best_sharpe:.2f}, Return: {best_metrics['annual_return']:.1%}, "
          f"MaxDD: {best_metrics['max_drawdown']:.1%}")
    print("\nBest Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

def main():
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_folder")
    data_dict = load_data(data_folder)
    if not data_dict:
        print("No data loaded! Check data folder path and file contents.")
        return
    
    print("\nOptimized MA Parameters:")
    for k, v in OPTIMIZED_PARAMS.items():
        print(f"  {k}: {v}")
    
    print("\nTrading Parameter Space:")
    total_combos = 1
    for param, values in TRADING_GRID.items():
        print(f"  {param}: {len(values)} values = {values}")
        total_combos *= len(values)
    print(f"\nTesting {total_combos} parameter combinations...")
    
    main_optimization(data_dict, max_runs=total_combos)

if __name__ == "__main__":
    main()
