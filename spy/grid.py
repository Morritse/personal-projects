import pandas as pd
import numpy as np
from itertools import product

# Assume these come from your existing code:
# run_momentum_strategy(df, params) -> returns {'TotalPnL':..., 'Sharpe':...}
# run_mean_reversion_strategy(df, params) -> ...
# run_breakout_strategy(df, params) -> ...

from backtest import run_momentum_strategy, run_mean_reversion_strategy, run_breakout_strategy

# Example param grids (define or import them)
momentum_param_grid = {
    "macd_fast": [8, 12, 15],
    "macd_slow": [17, 26, 30],
    "macd_signal": [9],
    "rsi_period": [7, 14],
    "rsi_overbought": [65, 70],
    "rsi_oversold": [30, 35]
}

meanrev_param_grid = {
    "boll_period": [10, 20],
    "boll_std": [2, 2.5],
    "rsi_period": [7, 14],
    "rsi_overbought": [70, 75],
    "rsi_oversold": [25, 30]
}

breakout_param_grid = {
    "donchian_lookback": [10, 14, 20, 30, 55]
}

def parameter_combinations(param_grid):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def optimize_momentum():
    df_5m = pd.read_csv("spy_5m_cleaned.csv", parse_dates=["timestamp"], index_col="timestamp")
    
    best_sharpe = float("-inf")
    best_params = None
    best_result = None
    
    for combo in parameter_combinations(momentum_param_grid):
        results = run_momentum_strategy(df_5m.copy(), combo)
        sharpe = results["Sharpe"]
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = combo
            best_result = results
    
    print("Best Momentum Params:", best_params)
    print("Best Momentum Result:", best_result)


def optimize_meanrev():
    df_15m = pd.read_csv("spy_15m_cleaned.csv", parse_dates=["timestamp"], index_col="timestamp")
    
    best_sharpe = float("-inf")
    best_params = None
    best_result = None
    
    for combo in parameter_combinations(meanrev_param_grid):
        results = run_mean_reversion_strategy(df_15m.copy(), combo)
        sharpe = results["Sharpe"]
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = combo
            best_result = results
    
    print("Best MeanReversion Params:", best_params)
    print("Best MeanReversion Result:", best_result)

def optimize_breakout():
    df_5m = pd.read_csv("spy_5m_cleaned.csv", parse_dates=["timestamp"], index_col="timestamp")
    
    best_sharpe = float("-inf")
    best_params = None
    best_result = None
    
    for combo in parameter_combinations(breakout_param_grid):
        results = run_breakout_strategy(df_5m.copy(), combo)
        sharpe = results["Sharpe"]
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = combo
            best_result = results
    
    print("Best Breakout Params:", best_params)
    print("Best Breakout Result:", best_result)


def main():
    print("Optimizing Momentum Strategy...")
    optimize_momentum()
    
    print("\nOptimizing MeanReversion Strategy...")
    optimize_meanrev()
    
    print("\nOptimizing Breakout Strategy...")
    optimize_breakout()

if __name__ == "__main__":
    main()
