import json
import pandas as pd
import numpy as np

###############################################################################
# Simple Indicator Functions
###############################################################################
def compute_macd(df, fast, slow, signal):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD_line'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD_line'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD_line'] - df['MACD_signal']
    return df

def compute_rsi(df, period):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).abs()
    loss = (delta.where(delta < 0, 0)).abs()

    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def compute_bollinger(df, period, std_dev):
    df['BB_mid'] = df['close'].rolling(period).mean()
    df['BB_std'] = df['close'].rolling(period).std()
    df['BB_upper'] = df['BB_mid'] + std_dev * df['BB_std']
    df['BB_lower'] = df['BB_mid'] - std_dev * df['BB_std']
    return df

def compute_donchian(df, lookback):
    df['donchian_high'] = df['high'].rolling(lookback).max()
    df['donchian_low'] = df['low'].rolling(lookback).min()
    return df


###############################################################################
# Helper: Calculate a simple "trade-based" Sharpe ratio
###############################################################################
def calculate_sharpe(trade_pnls):
    """
    A basic trade-based Sharpe ratio:
        Sharpe = mean(tradePnL) / stdev(tradePnL)
    This ignores time entirely (i.e., how long each trade took).
    
    If you want a time-based Sharpe, you'd need daily or bar-based returns.
    """
    if len(trade_pnls) < 2:
        return 0.0  # Not enough trades to get a meaningful std dev
    mean_pnl = np.mean(trade_pnls)
    std_pnl = np.std(trade_pnls, ddof=1)  # ddof=1 => sample stdev
    if std_pnl == 0:
        return 0.0
    return mean_pnl / std_pnl


###############################################################################
# Strategy Logic
###############################################################################
def run_momentum_strategy(df, params):
    macd_fast = params['macd_fast']
    macd_slow = params['macd_slow']
    macd_signal = params['macd_signal']
    rsi_period = params['rsi_period']
    rsi_overbought = params['rsi_overbought']
    rsi_oversold = params['rsi_oversold']

    df = compute_macd(df, macd_fast, macd_slow, macd_signal)
    df = compute_rsi(df, rsi_period)

    position = 0
    entry_price = 0
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        macd_line = row['MACD_line']
        macd_sig = row['MACD_signal']
        rsi_val = row['RSI']
        close_price = row['close']

        if pd.isna(macd_line) or pd.isna(macd_sig) or pd.isna(rsi_val):
            continue

        if position == 0:
            if (macd_line > macd_sig) and (rsi_val < rsi_overbought):
                position = 1
                entry_price = close_price
            elif (macd_line < macd_sig) and (rsi_val > rsi_oversold):
                position = -1
                entry_price = close_price
        else:
            if position == 1:
                if (macd_line < macd_sig) or (rsi_val > rsi_overbought):
                    pnl = close_price - entry_price
                    trades.append(pnl)
                    position = 0
            elif position == -1:
                if (macd_line > macd_sig) or (rsi_val < rsi_oversold):
                    pnl = entry_price - close_price
                    trades.append(pnl)
                    position = 0

    # Force close any open position at the last bar
    if position != 0:
        last_price = df['close'].iloc[-1]
        if position == 1:
            trades.append(last_price - entry_price)
        else:
            trades.append(entry_price - last_price)

    total_pnl = sum(trades)
    num_trades = len(trades)
    avg_pnl = total_pnl / num_trades if num_trades else 0
    win_rate = sum(1 for t in trades if t > 0) / num_trades if num_trades else 0
    sharpe_ratio = calculate_sharpe(trades)

    results = {
        'TotalPnL': total_pnl,
        'NumTrades': num_trades,
        'AvgPnL': avg_pnl,
        'WinRate': win_rate,
        'Sharpe': sharpe_ratio
    }
    return results


def run_mean_reversion_strategy(df, params):
    boll_period = params['boll_period']
    boll_std = params['boll_std']
    rsi_period = params['rsi_period']
    rsi_overbought = params['rsi_overbought']
    rsi_oversold = params['rsi_oversold']

    df = compute_bollinger(df, boll_period, boll_std)
    df = compute_rsi(df, rsi_period)

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

        if pd.isna(bb_lower) or pd.isna(bb_upper) or pd.isna(rsi_val):
            continue

        if position == 0:
            if (close_price <= bb_lower) and (rsi_val < rsi_oversold):
                position = 1
                entry_price = close_price
            elif (close_price >= bb_upper) and (rsi_val > rsi_overbought):
                position = -1
                entry_price = close_price
        else:
            if position == 1:
                if (close_price >= bb_mid) or (close_price >= bb_upper):
                    pnl = close_price - entry_price
                    trades.append(pnl)
                    position = 0
            elif position == -1:
                if (close_price <= bb_mid) or (close_price <= bb_lower):
                    pnl = entry_price - close_price
                    trades.append(pnl)
                    position = 0

    if position != 0:
        last_price = df['close'].iloc[-1]
        if position == 1:
            trades.append(last_price - entry_price)
        else:
            trades.append(entry_price - last_price)

    total_pnl = sum(trades)
    num_trades = len(trades)
    avg_pnl = total_pnl / num_trades if num_trades else 0
    win_rate = sum(1 for t in trades if t > 0) / num_trades if num_trades else 0
    sharpe_ratio = calculate_sharpe(trades)

    results = {
        'TotalPnL': total_pnl,
        'NumTrades': num_trades,
        'AvgPnL': avg_pnl,
        'WinRate': win_rate,
        'Sharpe': sharpe_ratio
    }
    return results


def run_breakout_strategy(df, params):
    lookback = params['donchian_lookback']
    df = compute_donchian(df, lookback)

    position = 0
    entry_price = 0
    trades = []
    prev_d_high = None
    prev_d_low = None

    for i in range(len(df)):
        row = df.iloc[i]
        close_price = row['close']
        d_high = row['donchian_high']
        d_low = row['donchian_low']

        if pd.isna(d_high) or pd.isna(d_low):
            continue

        # Store previous channel values for breakout confirmation
        if prev_d_high is None:
            prev_d_high = d_high
            prev_d_low = d_low
            continue

        if position == 0:
            # Enter long if we break above the previous high
            if close_price > prev_d_high:
                position = 1
                entry_price = close_price
            # Enter short if we break below the previous low
            elif close_price < prev_d_low:
                position = -1
                entry_price = close_price
        else:
            channel_middle = (d_high + d_low) / 2
            
            if position == 1:
                # Exit long if we close below the channel middle or make new low
                if close_price < channel_middle or close_price < d_low:
                    pnl = close_price - entry_price
                    trades.append(pnl)
                    position = 0
            elif position == -1:
                # Exit short if we close above the channel middle or make new high
                if close_price > channel_middle or close_price > d_high:
                    pnl = entry_price - close_price
                    trades.append(pnl)
                    position = 0

        prev_d_high = d_high
        prev_d_low = d_low

    if position != 0:
        last_price = df['close'].iloc[-1]
        if position == 1:
            trades.append(last_price - entry_price)
        else:
            trades.append(entry_price - last_price)

    total_pnl = sum(trades)
    num_trades = len(trades)
    avg_pnl = total_pnl / num_trades if num_trades else 0
    win_rate = sum(1 for t in trades if t > 0) / num_trades if num_trades else 0
    sharpe_ratio = calculate_sharpe(trades)

    results = {
        'TotalPnL': total_pnl,
        'NumTrades': num_trades,
        'AvgPnL': avg_pnl,
        'WinRate': win_rate,
        'Sharpe': sharpe_ratio
    }
    return results


###############################################################################
# Main Backtesting Flow
###############################################################################
def main():
    # 1) Load config
    with open("config.json", "r") as f:
        config = json.load(f)

    strategies = config["strategies"]

    for strat_name, strat_params in strategies.items():
        strat_type = strat_params["type"]
        timeframe = strat_params["timeframe"]

        filename = f"spy_{timeframe}_cleaned.csv"
        df = pd.read_csv(filename, parse_dates=["timestamp"], index_col="timestamp")

        if strat_type == "momentum":
            results = run_momentum_strategy(df, strat_params)
        elif strat_type == "mean_reversion":
            results = run_mean_reversion_strategy(df, strat_params)
        elif strat_type == "breakout":
            results = run_breakout_strategy(df, strat_params)
        else:
            print(f"[WARNING] Unknown strategy type: {strat_type}")
            continue

        print(f"--- Strategy: {strat_name} ---")
        for k, v in results.items():
            print(f"{k}: {v}")
        print("")

if __name__ == "__main__":
    main()
