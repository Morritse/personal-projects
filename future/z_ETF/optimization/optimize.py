import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import itertools
import datetime

# 1) Import your parameter grid from param_grid
from param_grid import PARAM_GRID

# ------------------------------------------------------------------
# Helper: Load your CSV data from a folder
# ------------------------------------------------------------------
def load_data(data_folder: str) -> Dict[str, pd.DataFrame]:
    """
    Loads CSV files from `data_folder`. Each CSV is assumed to be
    named something like `SPY.csv` and has columns:
        Date,open,high,low,close,volume
    Returns a dict: { symbol: df }, where df is a daily OHLCV DataFrame.
    """
    data_dict = {}

    # Match all .csv files in the folder
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    for path in csv_files:
        filename = os.path.basename(path)
        # e.g. "SPY.csv" -> "SPY"
        symbol = filename.replace(".csv", "")
        
        try:
            df = pd.read_csv(path)
            # Convert 'Date' to datetime and localize timezone
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            # Sort and set as index
            df.sort_values("Date", inplace=True)
            df.set_index("Date", inplace=True)
            
            # Basic cleaning
            df = df[['open','high','low','close','volume']].copy()
            df.dropna(subset=['open','high','low','close'], inplace=True)
            
            # Ensure numeric columns
            for col in ['open','high','low','close','volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows that became NaN after conversion
            df.dropna(inplace=True)

            # Print data quality info
            print(f"\nLoaded {symbol}:")
            print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"  Rows: {len(df)}")
            print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            data_dict[symbol] = df
            
        except Exception as e:
            print(f"Error loading {symbol}: {str(e)}")
            continue
    
    return data_dict

# ------------------------------------------------------------------
# Helper: Calculate daily returns, etc.
# ------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Given a DataFrame of daily OHLCV data and a set of parameters,
    compute necessary indicators: fast MA, slow MA, ATR, ADX, etc.
    Return a DataFrame with these columns appended.
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Unpack needed params
    lookback_fast = params['lookback_fast']
    lookback_slow = params['lookback_slow']
    vol_lookback  = params['vol_lookback']
    
    # Daily returns (handle inf/nan) - avoid chained assignment warning
    returns = df['close'].pct_change()
    df['returns'] = returns.replace([np.inf, -np.inf], np.nan)
    
    # Simple moving averages with min_periods
    df['ma_fast'] = df['close'].rolling(window=lookback_fast, min_periods=1).mean()
    df['ma_slow'] = df['close'].rolling(window=lookback_slow, min_periods=1).mean()
    
    # ATR calculation
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']  # Current high - low
    tr2 = (df['high'] - prev_close).abs()  # Current high - prev close
    tr3 = (df['low'] - prev_close).abs()  # Current low - prev close
    
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(window=vol_lookback, min_periods=1).mean()

    # Volatility calculation with min_periods
    df['roll_std'] = df['returns'].rolling(window=vol_lookback, min_periods=1).std() * np.sqrt(252)
    df['roll_std'] = df['roll_std'].replace(0, np.nan)  # Replace 0s with NaN
    
    # ADX calculation (simplified)
    if params['adx_threshold'] is not None:
        # +DM and -DM
        high_diff = df['high'] - df['high'].shift(1)
        low_diff = df['low'].shift(1) - df['low']
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Smooth with vol_lookback
        tr_ma = df['tr'].rolling(window=vol_lookback, min_periods=1).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=vol_lookback, min_periods=1).mean() / tr_ma
        minus_di = 100 * pd.Series(minus_dm).rolling(window=vol_lookback, min_periods=1).mean() / tr_ma
        
        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=vol_lookback, min_periods=1).mean()
    else:
        df['adx'] = 25  # default if not using ADX filter

    return df

# ------------------------------------------------------------------
# Strategy / Backtest
# ------------------------------------------------------------------
def backtest_symbol(df: pd.DataFrame, params: Dict[str, Any], verbose=True) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Single-symbol backtest using daily data in `df`, returns series
    of daily strategy returns and the full DataFrame with signals.
    """
    if len(df) < max(params['lookback_fast'], params['lookback_slow']):
        # Not enough data
        return pd.Series(0, index=df.index), df

    # Build signals or position
    # A simple dual-MA approach: 
    #   signal = 1 if ma_fast > ma_slow else -1 (or 0).
    # This is a demonstration; adapt your real logic.
    df['signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
    df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1

    # Optional ADX filter
    if params['adx_threshold'] is not None:
        # Zero out signals where ADX < threshold
        mask_low_adx = df['adx'] < params['adx_threshold']
        df.loc[mask_low_adx, 'signal'] = 0

    # Position sizing with volatility targeting
    vol_target = params['vol_target']
    df['pos_size'] = vol_target / (df['roll_std'] + 1e-9)  # avoid div by zero
    df['pos_size'] = df['pos_size'].clip(upper=1.0)
    
    # Final position = signal * size
    df['position'] = df['signal'] * df['pos_size']
    
    # Shift by 1 to simulate EOD or next-day open execution
    df['position_shifted'] = df['position'].shift(1).fillna(0)

    # Strategy daily returns
    df['strategy_ret'] = df['position_shifted'] * df['returns']
    
    if verbose:
        # Minimal output in verbose mode
        pass

    return df['strategy_ret'], df

def run_backtest_on_universe(
    data_dict: Dict[str, pd.DataFrame],
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    For each symbol in `data_dict`, run backtest, return DataFrame of 
    daily returns per symbol, plus 'portfolio'.
    """
    all_rets = pd.DataFrame()
    
    for symbol, df_raw in data_dict.items():
        # Silently process each symbol
        df = compute_indicators(df_raw.copy(), params)
        daily_strat, _ = backtest_symbol(df, params, verbose=False)
        daily_strat.name = symbol
        all_rets = pd.concat([all_rets, daily_strat], axis=1)
    
    # Drop rows where all are NaN
    all_rets.dropna(how='all', inplace=True)
    
    # Simple equally-weighted average across symbols 
    all_rets['portfolio'] = all_rets.mean(axis=1)
    return all_rets

def compute_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Compute basic performance metrics from daily returns of the portfolio.
    """
    ann_factor = 252  # daily data

    # drop NaN
    rets = returns.dropna()
    if len(rets) < 2:
        return {
            "annual_return": 0,
            "annual_vol": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0
        }
    
    # Annual Return
    mean_daily = rets.mean()
    annual_ret = mean_daily * ann_factor

    # Annual Vol
    daily_std = rets.std()
    annual_vol = daily_std * np.sqrt(ann_factor)

    # Sharpe
    if annual_vol > 1e-9:
        sharpe = annual_ret / annual_vol
    else:
        sharpe = 0

    # Max Drawdown
    #   cumulate, find peak-to-trough
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1).min()

    return {
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": dd
    }

# ------------------------------------------------------------------
# Simple grid optimization (or you can do a GA approach)
# ------------------------------------------------------------------
def param_combinations(param_grid: Dict[str, List[Any]]):
    """
    Generate all combinations from the param grid, or adapt for your GA.
    """
    # Get only the parameters we use
    used_params = {
        "lookback_fast": param_grid["lookback_fast"],
        "lookback_slow": param_grid["lookback_slow"],
        "vol_lookback": param_grid["vol_lookback"],
        "vol_target": param_grid["vol_target"],
        "adx_threshold": param_grid["adx_threshold"]
    }
    
    keys = list(used_params.keys())
    # Cartesian product
    for combo_vals in itertools.product(*(used_params[k] for k in keys)):
        yield dict(zip(keys, combo_vals))

def main_optimization(data_dict: Dict[str, pd.DataFrame], max_runs=1000):
    """
    Runs a naive cartesian-grid search over the param grid, 
    or else adapt it for a random/GAs approach if the param space is large.
    """
    best_sharpe = -999
    best_params = None
    best_metrics = {}
    best_returns = None

    run_count = 0
    for param_set in param_combinations(PARAM_GRID):
        run_count += 1
        if run_count > max_runs:
            # safety break if you want to limit 
            break
        
        # 1) Run backtest
        all_rets = run_backtest_on_universe(data_dict, param_set)
        
        # 2) Evaluate on portfolio
        portfolio_rets = all_rets['portfolio']
        metrics = compute_metrics(portfolio_rets)
        
        # Print current run results and key parameters
        print(f"\nRun {run_count}:")
        print(f"  Params: fast={param_set['lookback_fast']}, slow={param_set['lookback_slow']}, vol={param_set['vol_lookback']}, target={param_set['vol_target']}, adx={param_set['adx_threshold']}")
        print(f"  Results: Sharpe: {metrics['sharpe_ratio']:.2f}, Return: {metrics['annual_return']:.1%}, MaxDD: {metrics['max_drawdown']:.1%}")
        
        # 3) Track best
        if metrics['sharpe_ratio'] > best_sharpe:
            best_sharpe = metrics['sharpe_ratio']
            best_params = param_set
            best_metrics = metrics
            best_returns = all_rets
    
    print("\n=== Best Result ===")
    print(f"Sharpe: {best_sharpe:.2f}, Return: {best_metrics['annual_return']:.1%}, MaxDD: {best_metrics['max_drawdown']:.1%}")
    print("\nBest Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

def main():
    # Use absolute path
    data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_folder")
    data_dict = load_data(data_folder)
    if not data_dict:
        print("No data loaded! Check data folder path and file contents.")
        return
    
    # Calculate total combinations for used parameters only
    used_params = {
        "lookback_fast": PARAM_GRID["lookback_fast"],
        "lookback_slow": PARAM_GRID["lookback_slow"],
        "vol_lookback": PARAM_GRID["vol_lookback"],
        "vol_target": PARAM_GRID["vol_target"],
        "adx_threshold": PARAM_GRID["adx_threshold"]
    }
    
    # Print parameter space
    print("\nParameter Space:")
    total_combos = 1
    for param, values in used_params.items():
        print(f"  {param}: {len(values)} values = {values}")
        total_combos *= len(values)
    print(f"\nTesting {total_combos} parameter combinations...")
    
    main_optimization(data_dict, max_runs=total_combos)

if __name__ == "__main__":
    main()
