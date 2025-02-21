#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
from statistics import stdev, mean

###############################################################################
# Indicator Functions
###############################################################################
def compute_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD_line'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD_line'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
    return df

def compute_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).abs()
    loss = (delta.where(delta < 0, 0)).abs()

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def compute_bollinger(df, period=20, std_dev=2):
    df['BB_mid'] = df['close'].rolling(period).mean()
    df['BB_std'] = df['close'].rolling(period).std()
    df['BB_upper'] = df['BB_mid'] + std_dev * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - std_dev * df['BB_std']
    return df

def compute_donchian(df, lookback=20):
    df['donchian_high'] = df['high'].rolling(lookback).max()
    df['donchian_low']  = df['low'].rolling(lookback).min()
    return df

###############################################################################
# Simple "trade-based" Sharpe ratio
###############################################################################
def calculate_sharpe(trade_pnls):
    """
    A basic trade-based Sharpe ratio:
       Sharpe = mean(tradePnL) / stdev(tradePnL)
    ignoring time durations.
    """
    if len(trade_pnls) < 2:
        return 0.0
    m = mean(trade_pnls)
    s = stdev(trade_pnls)
    return m / s if s != 0 else 0.0

###############################################################################
# Strategy "States" for Parallel
###############################################################################
class StrategyState:
    def __init__(self, name, initial_capital=10000.0):
        self.name = name
        self.position = 0            # 1 = long, -1 = short, 0 = flat
        self.entry_price = 0.0
        self.trades = []            # store all PnLs
        self.capital = initial_capital  # optional, if you want sub-account logic
        self.realized_pnl = 0.0

    def __repr__(self):
        return (f"<StrategyState {self.name}, pos={self.position}, "
                f"realized={self.realized_pnl:.2f}, capital={self.capital:.2f}>")

###############################################################################
# Single-Bar Update Logic for Each Strategy
###############################################################################
def momentum_step(state: StrategyState, row, params):
    """
    Momentum bar-by-bar logic with example:
      - stop_loss_pct
      - trailing_stop_pct
      - risk-based position sizing
    """
    macd_line = row['MACD_line']
    macd_sig  = row['MACD_signal']
    rsi_val   = row['RSI']
    close_px  = row['close']

    stop_loss_pct     = params.get('stop_loss_pct', 0.01)       # 1% stop
    trailing_stop_pct = params.get('trailing_stop_pct', 0.02)   # 2% trailing
    rsi_overbought    = params['rsi_overbought']
    rsi_oversold      = params['rsi_oversold']

    # Position sizing - e.g., risk 1% of sub-account capital
    risk_pct_of_capital = params.get('risk_pct_of_capital', 0.01)  # 1%
    # If we get a signal to go long, we might buy enough shares so that if we lose stop_loss_pct,
    # it equals 1% of capital, e.g.:
    # shares = (capital * risk_pct_of_capital) / (close_px * stop_loss_pct)

    if state.position == 0:
        # Potential entries
        if (macd_line > macd_sig) and (rsi_val < rsi_overbought):
            # enter long
            shares = (state.capital * risk_pct_of_capital) / (close_px * stop_loss_pct)
            shares = int(shares)  # round down to whole shares

            state.position = shares
            state.entry_price = close_px
            # store a 'stop_price' if you want a static stop (like close_px * (1 - stop_loss_pct))
            state.stop_price = close_px * (1 - stop_loss_pct)
            # initialize trailing stop
            state.trailing_high = close_px

        elif (macd_line < macd_sig) and (rsi_val > rsi_oversold):
            # enter short
            shares = (state.capital * risk_pct_of_capital) / (close_px * stop_loss_pct)
            shares = int(shares)

            state.position = -shares
            state.entry_price = close_px
            state.stop_price = close_px * (1 + stop_loss_pct)
            state.trailing_low = close_px

    else:
        # Already in a position
        shares_held = abs(state.position)
        entry = state.entry_price

        # If in a long
        if state.position > 0:
            # Update trailing stop?
            if close_px > state.trailing_high:
                state.trailing_high = close_px
            trailing_stop_price = state.trailing_high * (1 - trailing_stop_pct)

            # Check for exit conditions:
            # 1) momentum reversal
            if (macd_line < macd_sig) or (rsi_val > rsi_overbought):
                pnl = (close_px - entry) * shares_held
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0

            # 2) static stop-loss
            elif close_px < state.stop_price:
                pnl = (close_px - entry) * shares_held
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0

            # 3) trailing stop
            elif close_px < trailing_stop_price:
                pnl = (close_px - entry) * shares_held
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0

        # If in a short
        else:
            if close_px < state.trailing_low:
                state.trailing_low = close_px
            trailing_stop_price = state.trailing_low * (1 + trailing_stop_pct)

            if (macd_line > macd_sig) or (rsi_val < rsi_oversold):
                pnl = (entry - close_px) * shares_held
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0

            elif close_px > state.stop_price:
                pnl = (entry - close_px) * shares_held
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0

            elif close_px > trailing_stop_price:
                pnl = (entry - close_px) * shares_held
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0

def meanrev_step(state: StrategyState, row, params):
    """
    Single-bar update for Mean Reversion strategy.
    Expects row to have: 'BB_lower', 'BB_upper', 'BB_mid', 'RSI', 'close'.
    """
    close_price = row['close']
    bb_lower    = row['BB_lower']
    bb_upper    = row['BB_upper']
    bb_mid      = row['BB_mid']
    rsi_val     = row['RSI']

    rsi_overbought = params['rsi_overbought']
    rsi_oversold   = params['rsi_oversold']

    if state.position == 0:
        # Enter long if close < BB_lower & RSI < oversold
        if (close_price <= bb_lower) and (rsi_val < rsi_oversold):
            state.position = 1
            state.entry_price = close_price
        # Enter short if close > BB_upper & RSI > overbought
        elif (close_price >= bb_upper) and (rsi_val > rsi_overbought):
            state.position = -1
            state.entry_price = close_price
    else:
        if state.position == 1:
            # Exit if close >= mid-band or close >= BB_upper
            if (close_price >= bb_mid) or (close_price >= bb_upper):
                pnl = close_price - state.entry_price
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0
        elif state.position == -1:
            # Exit if close <= mid-band or close <= BB_lower
            if (close_price <= bb_mid) or (close_price <= bb_lower):
                pnl = state.entry_price - close_price
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0

def breakout_step(state: StrategyState, current_row, prev_row, params):
    """
    Single-bar update for Breakout strategy using Donchian & channel middle exit logic.
    `current_row` has: 'donchian_high', 'donchian_low', 'close'
    `prev_row` is the bar from the prior step (for previous channel).
    """
    close_price = current_row['close']
    d_high      = current_row['donchian_high']
    d_low       = current_row['donchian_low']

    # Or we can reference the "previous bar" high/low if you want a 1-bar-lag breakout:
    # but your example code uses a 'prev_d_high' approach. We'll do a slight adaptation:
    if (prev_row is None) or pd.isna(prev_row['donchian_high']) or pd.isna(prev_row['donchian_low']):
        return  # skip first bar or if NaN

    prev_d_high = prev_row['donchian_high']
    prev_d_low  = prev_row['donchian_low']

    channel_middle = (d_high + d_low) / 2

    if state.position == 0:
        # Enter if close > prev_d_high => long
        if close_price > prev_d_high:
            state.position = 1
            state.entry_price = close_price
        # Enter if close < prev_d_low => short
        elif close_price < prev_d_low:
            state.position = -1
            state.entry_price = close_price
    else:
        if state.position == 1:
            # exit long if close < channel_middle or close < d_low
            if (close_price < channel_middle) or (close_price < d_low):
                pnl = close_price - state.entry_price
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0
        elif state.position == -1:
            if (close_price > channel_middle) or (close_price > d_high):
                pnl = state.entry_price - close_price
                state.realized_pnl += pnl
                state.trades.append(pnl)
                state.position = 0


###############################################################################
# Parallel Backtester
###############################################################################

def run_parallel_backtest(df_5m, df_15m, 
                          momentum_params, meanrev_params, breakout_params,
                          initial_capital=10000.0):
    """
    Creates three StrategyStates (momentum, meanrev, breakout), 
    runs them in parallel over the dataset. 
    Summarizes each sub-strategy's PnL, plus combined.
    """

    # 1) Pre-compute indicators for momentum (MACD, RSI) in df_5m
    df_5m = compute_macd(df_5m, 
                         fast=momentum_params['macd_fast'], 
                         slow=momentum_params['macd_slow'], 
                         signal=momentum_params['macd_signal'])
    df_5m = compute_rsi(df_5m, period=momentum_params['rsi_period'])

    # 2) Pre-compute indicators for mean reversion (Boll+RSI) in df_15m
    df_15m = compute_bollinger(df_15m, 
                               period=meanrev_params['boll_period'], 
                               std_dev=meanrev_params['boll_std'])
    df_15m = compute_rsi(df_15m, period=meanrev_params['rsi_period'])

    # 3) Pre-compute indicators for breakout (Donchian) in df_5m
    df_5m = compute_donchian(df_5m, 
                             lookback=breakout_params['donchian_lookback'])

    # 4) Initialize sub-states
    momentum_state = StrategyState("Momentum", initial_capital)
    meanrev_state  = StrategyState("MeanReversion", initial_capital)
    breakout_state = StrategyState("Breakout", initial_capital)

    # 5) Union of timestamps from 5m and 15m
    all_timestamps = sorted(list(set(df_5m.index).union(df_15m.index)))
    last_row_5m = None
    last_row_15m = None

    for ts in all_timestamps:
        row_5m = df_5m.loc[ts] if ts in df_5m.index else None
        row_15m = df_15m.loc[ts] if ts in df_15m.index else None

        # Momentum & Breakout => 5m data
        if row_5m is not None:
            # Single-bar update for momentum
            momentum_step(momentum_state, row_5m, momentum_params)

            # Single-bar update for breakout (needs prev_row)
            if last_row_5m is not None:
                breakout_step(breakout_state, row_5m, last_row_5m, breakout_params)
            last_row_5m = row_5m

        # Mean Reversion => 15m data
        if row_15m is not None:
            meanrev_step(meanrev_state, row_15m, meanrev_params)
            last_row_15m = row_15m

    # 6) Force-close any open position
    final_close_5m = df_5m["close"].iloc[-1]
    final_close_15m = df_15m["close"].iloc[-1]

    # For momentum & breakout, we use final_close_5m
    if momentum_state.position != 0:
        if momentum_state.position == 1:
            pnl = final_close_5m - momentum_state.entry_price
        else:
            pnl = momentum_state.entry_price - final_close_5m
        momentum_state.realized_pnl += pnl
        momentum_state.trades.append(pnl)
        momentum_state.position = 0

    if breakout_state.position != 0:
        if breakout_state.position == 1:
            pnl = final_close_5m - breakout_state.entry_price
        else:
            pnl = breakout_state.entry_price - final_close_5m
        breakout_state.realized_pnl += pnl
        breakout_state.trades.append(pnl)
        breakout_state.position = 0

    # For mean reversion, we use final_close_15m
    if meanrev_state.position != 0:
        if meanrev_state.position == 1:
            pnl = final_close_15m - meanrev_state.entry_price
        else:
            pnl = meanrev_state.entry_price - final_close_15m
        meanrev_state.realized_pnl += pnl
        meanrev_state.trades.append(pnl)
        meanrev_state.position = 0

    # 7) Summarize results
    total_pnl_m = momentum_state.realized_pnl
    total_pnl_r = meanrev_state.realized_pnl
    total_pnl_b = breakout_state.realized_pnl

    combined_pnl = total_pnl_m + total_pnl_r + total_pnl_b

    sharpe_m = calculate_sharpe(momentum_state.trades)
    sharpe_r = calculate_sharpe(meanrev_state.trades)
    sharpe_b = calculate_sharpe(breakout_state.trades)

    results = {
        "Momentum": {
            "TotalPnL": total_pnl_m,
            "NumTrades": len(momentum_state.trades),
            "Sharpe": sharpe_m
        },
        "MeanReversion": {
            "TotalPnL": total_pnl_r,
            "NumTrades": len(meanrev_state.trades),
            "Sharpe": sharpe_r
        },
        "Breakout": {
            "TotalPnL": total_pnl_b,
            "NumTrades": len(breakout_state.trades),
            "Sharpe": sharpe_b
        },
        "EnsembleTotalPnL": combined_pnl
    }
    return results

###############################################################################
# Main Entry
###############################################################################
def main():
    # 1) Load config with final params
    with open("config.json", "r") as f:
        config = json.load(f)

    # Example config structure:
    # {
    #   "strategies": {
    #       "momentum_5m": {...},
    #       "meanrev_15m": {...},
    #       "breakout_5m": {...}
    #   }
    # }

    # Extract param dicts from your config
    momentum_cfg   = config["strategies"]["momentum_5m"]   # e.g., "macd_fast", "macd_slow", etc.
    meanrev_cfg    = config["strategies"]["meanrev_15m"]   # e.g., "boll_period", "rsi_period", etc.
    breakout_cfg   = config["strategies"]["breakout_5m"]   # e.g., "donchian_lookback"

    # 2) Load the relevant CSV data
    #   e.g. "spy_5m_cleaned.csv", "spy_15m_cleaned.csv"
    file_5m = "spy_5m_cleaned.csv"
    file_15m = "spy_15m_cleaned.csv"

    df_5m = pd.read_csv(file_5m, parse_dates=["timestamp"], index_col="timestamp")
    df_15m = pd.read_csv(file_15m, parse_dates=["timestamp"], index_col="timestamp")

    # 3) Run the parallel backtest
    results = run_parallel_backtest(
        df_5m, df_15m,
        momentum_cfg, meanrev_cfg, breakout_cfg,
        initial_capital=10000.0
    )

    # 4) Print final results
    print("=== Parallel Backtest Results ===")
    for strat_name in ["Momentum", "MeanReversion", "Breakout"]:
        sub = results[strat_name]
        print(f"{strat_name}: TotalPnL={sub['TotalPnL']:.2f}, "
              f"NumTrades={sub['NumTrades']}, Sharpe={sub['Sharpe']:.4f}")

    print(f"Combined Ensemble PnL: {results['EnsembleTotalPnL']:.2f}")

if __name__ == "__main__":
    main()
