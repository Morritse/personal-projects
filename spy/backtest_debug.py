import json
import pandas as pd
import numpy as np

###############################################################################
# Simple Indicator Functions
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
    df['donchian_low'] = df['low'].rolling(lookback).min()
    return df


###############################################################################
# Strategy Logic with Debug
###############################################################################

def run_momentum_strategy(df, params, debug=False):
    """
    Example logic:
      - Long if MACD_line > MACD_signal and RSI < rsi_overbought
      - Short if MACD_line < MACD_signal and RSI > rsi_oversold
      - Exit when opposite condition is triggered or RSI crosses extremes.
    """

    macd_fast = params.get('macd_fast', 12)
    macd_slow = params.get('macd_slow', 26)
    macd_signal = params.get('macd_signal', 9)
    rsi_period = params.get('rsi_period', 14)
    rsi_overbought = params.get('rsi_overbought', 70)
    rsi_oversold = params.get('rsi_oversold', 30)

    df = compute_macd(df, macd_fast, macd_slow, macd_signal)
    df = compute_rsi(df, rsi_period)

    if debug:
        print(f"[DEBUG] Momentum Strategy: MACD params=({macd_fast},{macd_slow},{macd_signal}), RSI={rsi_period}")
        # Print first few rows of indicators to ensure they're not NaN
        print("[DEBUG] Sample of MACD_line, MACD_signal, RSI:")
        print(df[['MACD_line','MACD_signal','RSI']].head(5))

    position = 0  # 1 for long, -1 for short, 0 for flat
    entry_price = 0
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        macd_line = row['MACD_line']
        macd_sig = row['MACD_signal']
        rsi_val = row['RSI']
        close_price = row['close']

        if pd.isna(macd_line) or pd.isna(macd_sig) or pd.isna(rsi_val):
            continue  # Skip rows where indicators aren't ready

        if position == 0:
            if (macd_line > macd_sig) and (rsi_val < rsi_overbought):
                position = 1
                entry_price = close_price
                if debug:
                    print(f"[DEBUG] LONG ENTRY at index={df.index[i]} price={close_price:.2f} MACD={macd_line:.2f}>{macd_sig:.2f}, RSI={rsi_val:.2f}<={rsi_overbought}")
            elif (macd_line < macd_sig) and (rsi_val > rsi_oversold):
                position = -1
                entry_price = close_price
                if debug:
                    print(f"[DEBUG] SHORT ENTRY at index={df.index[i]} price={close_price:.2f} MACD={macd_line:.2f}<{macd_sig:.2f}, RSI={rsi_val:.2f}>={rsi_oversold}")
        else:
            # If in a long
            if position == 1:
                if (macd_line < macd_sig) or (rsi_val > rsi_overbought):
                    pnl = close_price - entry_price
                    trades.append(pnl)
                    position = 0
                    if debug:
                        print(f"[DEBUG] LONG EXIT at index={df.index[i]} price={close_price:.2f}, PnL={pnl:.2f}")
            # If in a short
            elif position == -1:
                if (macd_line > macd_sig) or (rsi_val < rsi_oversold):
                    pnl = entry_price - close_price
                    trades.append(pnl)
                    position = 0
                    if debug:
                        print(f"[DEBUG] SHORT EXIT at index={df.index[i]} price={close_price:.2f}, PnL={pnl:.2f}")

    # Close any open position at the end
    if position != 0:
        last_price = df['close'].iloc[-1]
        if position == 1:
            pnl = last_price - entry_price
            trades.append(pnl)
            if debug:
                print(f"[DEBUG] FORCED LONG EXIT at last bar: PnL={pnl:.2f}")
        else:
            pnl = entry_price - last_price
            trades.append(pnl)
            if debug:
                print(f"[DEBUG] FORCED SHORT EXIT at last bar: PnL={pnl:.2f}")

    total_pnl = sum(trades)
    num_trades = len(trades)
    avg_pnl = total_pnl / num_trades if num_trades else 0
    win_rate = sum(1 for t in trades if t > 0) / num_trades if num_trades else 0

    results = {
        'TotalPnL': total_pnl,
        'NumTrades': num_trades,
        'AvgPnL': avg_pnl,
        'WinRate': win_rate
    }
    return results


def run_mean_reversion_strategy(df, params, debug=False):
    """
    Example logic:
      - If close <= lower Boll band & RSI < oversold => go long
      - If close >= upper Boll band & RSI > overbought => go short
      - Exit at mid-band or partial revert, etc.
    """

    boll_period = params.get('boll_period', 20)
    boll_std = params.get('boll_std', 2)
    rsi_period = params.get('rsi_period', 14)
    rsi_overbought = params.get('rsi_overbought', 70)
    rsi_oversold = params.get('rsi_oversold', 30)

    df = compute_bollinger(df, boll_period, boll_std)
    df = compute_rsi(df, rsi_period)

    if debug:
        print(f"[DEBUG] MeanRev Strategy: Boll=({boll_period},{boll_std}), RSI={rsi_period}")
        print("[DEBUG] Sample of BB_upper, BB_lower, RSI:")
        print(df[['BB_upper','BB_lower','RSI']].head(5))

    position = 0
    entry_price = 0
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        close_price = row['close']
        bb_lower = row['BB_lower']
        bb_upper = row['BB_upper']
        bb_mid = row['BB_mid']
        rsi_val = row['RSI']

        # Skip if boll or rsi not ready
        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(rsi_val):
            continue

        if position == 0:
            # Go long
            if (close_price <= bb_lower) and (rsi_val < rsi_oversold):
                position = 1
                entry_price = close_price
                if debug:
                    print(f"[DEBUG] LONG ENTRY at {df.index[i]} close={close_price:.2f}, BLower={bb_lower:.2f}, RSI={rsi_val:.2f}")
            # Go short
            elif (close_price >= bb_upper) and (rsi_val > rsi_overbought):
                position = -1
                entry_price = close_price
                if debug:
                    print(f"[DEBUG] SHORT ENTRY at {df.index[i]} close={close_price:.2f}, BUpper={bb_upper:.2f}, RSI={rsi_val:.2f}")
        else:
            if position == 1:
                # Exit if price >= mid band or hits upper band
                if (close_price >= bb_mid) or (close_price >= bb_upper):
                    pnl = close_price - entry_price
                    trades.append(pnl)
                    position = 0
                    if debug:
                        print(f"[DEBUG] LONG EXIT at {df.index[i]}, close={close_price:.2f}, midBand={bb_mid:.2f}, PnL={pnl:.2f}")
            elif position == -1:
                # Exit if price <= mid band or hits lower band
                if (close_price <= bb_mid) or (close_price <= bb_lower):
                    pnl = entry_price - close_price
                    trades.append(pnl)
                    position = 0
                    if debug:
                        print(f"[DEBUG] SHORT EXIT at {df.index[i]}, close={close_price:.2f}, midBand={bb_mid:.2f}, PnL={pnl:.2f}")

    # Close any open position at the end
    if position != 0:
        last_price = df['close'].iloc[-1]
        if position == 1:
            pnl = last_price - entry_price
            trades.append(pnl)
            if debug:
                print(f"[DEBUG] FORCED LONG EXIT at last bar: PnL={pnl:.2f}")
        else:
            pnl = entry_price - last_price
            trades.append(pnl)
            if debug:
                print(f"[DEBUG] FORCED SHORT EXIT at last bar: PnL={pnl:.2f}")

    total_pnl = sum(trades)
    num_trades = len(trades)
    avg_pnl = total_pnl / num_trades if num_trades else 0
    win_rate = sum(1 for t in trades if t > 0) / num_trades if num_trades else 0

    results = {
        'TotalPnL': total_pnl,
        'NumTrades': num_trades,
        'AvgPnL': avg_pnl,
        'WinRate': win_rate
    }
    return results


def run_breakout_strategy(df, params, debug=False):
    """
    Example logic:
      - If close > donchian_high => go long
      - If close < donchian_low => go short
      - Exit if close crosses the opposite boundary, etc.
    """
    lookback = params.get('donchian_lookback', 20)
    df = compute_donchian(df, lookback)

    if debug:
        print(f"[DEBUG] Breakout Strategy: Donchian lookback={lookback}")
        print("[DEBUG] Sample of donchian_high, donchian_low:")
        print(df[['donchian_high','donchian_low','close']].head(5))

    position = 0
    entry_price = 0
    trades = []
    triggered_trade = False

    for i in range(len(df)):
        row = df.iloc[i]
        close_price = row['close']
        d_high = row['donchian_high']
        d_low = row['donchian_low']

        if pd.isna(d_high) or pd.isna(d_low):
            continue

        if position == 0:
            if close_price > d_high:
                position = 1
                entry_price = close_price
                triggered_trade = True
                if debug:
                    print(f"[DEBUG] LONG ENTRY at {df.index[i]} close={close_price:.2f}, dHigh={d_high:.2f}")
            elif close_price < d_low:
                position = -1
                entry_price = close_price
                triggered_trade = True
                if debug:
                    print(f"[DEBUG] SHORT ENTRY at {df.index[i]} close={close_price:.2f}, dLow={d_low:.2f}")
        else:
            if position == 1:
                if close_price < d_low:
                    pnl = close_price - entry_price
                    trades.append(pnl)
                    position = 0
                    if debug:
                        print(f"[DEBUG] LONG EXIT => reverse breakout at {df.index[i]}, close={close_price:.2f}, PnL={pnl:.2f}")
            elif position == -1:
                if close_price > d_high:
                    pnl = entry_price - close_price
                    trades.append(pnl)
                    position = 0
                    if debug:
                        print(f"[DEBUG] SHORT EXIT => reverse breakout at {df.index[i]}, close={close_price:.2f}, PnL={pnl:.2f}")

    # Close any open position at the end
    if position != 0:
        last_price = df['close'].iloc[-1]
        if position == 1:
            pnl = last_price - entry_price
            trades.append(pnl)
            if debug:
                print(f"[DEBUG] FORCED LONG EXIT at last bar: PnL={pnl:.2f}")
        else:
            pnl = entry_price - last_price
            trades.append(pnl)
            if debug:
                print(f"[DEBUG] FORCED SHORT EXIT at last bar: PnL={pnl:.2f}")

    if debug and not triggered_trade:
        print("[DEBUG] Breakout Strategy: NO TRADES TRIGGERED at all.")

    total_pnl = sum(trades)
    num_trades = len(trades)
    avg_pnl = total_pnl / num_trades if num_trades else 0
    win_rate = sum(1 for t in trades if t > 0) / num_trades if num_trades else 0

    results = {
        'TotalPnL': total_pnl,
        'NumTrades': num_trades,
        'AvgPnL': avg_pnl,
        'WinRate': win_rate
    }
    return results


###############################################################################
# Main Backtesting Flow
###############################################################################

def main(debug=False):
    """
    If debug=True, you'll see extra print statements about indicator values,
    trade signals, etc.
    """
    import os, sys
    # Load config from "config.json"
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("[ERROR] config.json not found. Exiting.")
        sys.exit(1)

    strategies = config.get("strategies", {})

    for strat_name, strat_params in strategies.items():
        strat_type = strat_params["type"]
        timeframe = strat_params["timeframe"]

        # Load the correct CSV for this timeframe
        filename = f"spy_{timeframe}_cleaned.csv"
        if not os.path.isfile(filename):
            print(f"[WARNING] File {filename} does not exist. Skipping.")
            continue

        df = pd.read_csv(filename, parse_dates=["timestamp"], index_col="timestamp")
        # Sort by index, just to be sure
        df.sort_index(inplace=True)

        if debug:
            print(f"\n=== Running Strategy: {strat_name} | type={strat_type} | timeframe={timeframe} ===")

        if strat_type == "momentum":
            results = run_momentum_strategy(df, strat_params, debug=debug)
        elif strat_type == "mean_reversion":
            results = run_mean_reversion_strategy(df, strat_params, debug=debug)
        elif strat_type == "breakout":
            results = run_breakout_strategy(df, strat_params, debug=debug)
        else:
            print(f"[WARNING] Unknown strategy type: {strat_type}")
            continue

        print(f"--- Strategy: {strat_name} ---")
        for k, v in results.items():
            print(f"{k}: {v}")
        print("")

if __name__ == "__main__":
    # Run with debug=True to see the statements
    main(debug=True)
