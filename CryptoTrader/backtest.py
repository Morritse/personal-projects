import pandas as pd
import numpy as np
from talib import abstract

#############################################################################
# 0) Indicators
#############################################################################
def compute_indicators_and_signal(
    df,
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    bb_period=20,
    bb_devup=2.0,
    bb_devdn=2.0
):
    """
    Compute RSI, MACD, Bollinger, then combine into 'combined_signal' in [-1..+1].
    """
    inputs = {
        'open': df['open'],
        'high': df['high'],
        'low': df['low'],
        'close': df['close'],
        'volume': df['volume']
    }

    # RSI
    df['rsi'] = abstract.Function('RSI')(inputs, timeperiod=rsi_period)

    # MACD
    macd_func = abstract.Function('MACD')
    macd_out = macd_func(
        inputs, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal
    )
    df['macd'], df['macd_signal'], df['macd_hist'] = macd_out

    # Bollinger Bands
    bb_func = abstract.Function('BBANDS')
    bb_out = bb_func(
        inputs,
        timeperiod=bb_period,
        nbdevup=bb_devup,
        nbdevdn=bb_devdn,
        matype=0
    )
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = bb_out

    df.dropna(inplace=True)

    # Normalize RSI: [0..100] -> [-1..+1]
    df['rsi_norm'] = 2 * ((df['rsi'] - 50) / 100)

    # MACD histogram => tanh
    scaling_factor = df['macd_hist'].std() or 1e-9
    df['macd_hist_norm'] = np.tanh(df['macd_hist'] / scaling_factor)

    # Bollinger %b => [-1..+1]
    df['bb_percent_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_norm'] = 2.0 * (df['bb_percent_b'] - 0.5)

    # Combine equally
    df['raw_signal'] = df['rsi_norm'] + df['macd_hist_norm'] + df['bb_norm']
    df['combined_signal'] = df['raw_signal'] / 3.0

    return df

#############################################################################
# 1) Generate Trade Signal with Threshold + Hysteresis (optional)
#############################################################################
def generate_trade_signal(row, threshold_up=0.8, threshold_down=-0.8):
    """
    Convert a continuous combined_signal into a discrete trade signal:
      +1 for long, -1 for short, 0 for flat.
    """
    if row['combined_signal'] > threshold_up:
        return 1   # long
    elif row['combined_signal'] < threshold_down:
        return -1  # short
    else:
        return 0   # flat

#############################################################################
# 2) Backtest with Stop-Loss, Position Sizing, Cooldown, etc.
#############################################################################
def run_backtest_on_df(
    df,
    initial_capital=10000.0,
    fee_rate=0.0005,
    slippage_rate=0.0002,
    max_risk_percent=0.02,
    stop_loss_frac=0.02,
    cooldown_bars=10,
    max_bars_in_trade=None,
):
    """
    Extended backtest with position sizing, stop-loss, cooldown, etc.
    Returns:
      - trades (list of dicts)
      - total_pnl (float)
      - pct_return (float)
      - daily_equity (pd.Series) : equity curve at each bar
    """
    trades = []
    position_side = 0   # +1 (long), -1 (short), or 0 (flat)
    position_size = 0.0
    entry_price = 0.0
    equity = initial_capital

    df = df.copy()
    df['signal'] = 0
    df['pnl'] = 0.0
    df['next_open'] = df['open'].shift(-1)  # we enter/exit at next bar's open

    # We will store an 'equity_curve' to track equity bar-by-bar
    equity_curve = [equity]

    # Generate discrete signals
    df['signal'] = df.apply(generate_trade_signal, axis=1)

    last_entry_idx = -999
    position_entry_idx = None

    for i in range(len(df)):
        if i == len(df) - 1:
            # Last bar: update equity curve and exit loop
            equity_curve.append(equity)
            break

        current_signal = df.iloc[i]['signal']
        current_open = df.iloc[i]['open']
        next_open = df.iloc[i]['next_open']
        if np.isnan(next_open):
            next_open = current_open

        # If we have a position, check for time exit or stop-loss
        if position_side != 0:
            if max_bars_in_trade is not None and (i - position_entry_idx >= max_bars_in_trade):
                # Exit on next bar
                trade_price = next_open
                if position_side > 0:
                    trade_price *= (1 - slippage_rate)
                    pnl_points = trade_price - entry_price
                else:
                    trade_price *= (1 + slippage_rate)
                    pnl_points = entry_price - trade_price

                exit_fee = trade_price * fee_rate * abs(position_size)
                net_pnl = (pnl_points * position_size) - exit_fee
                equity += net_pnl

                trades[-1]['exit_time'] = df.index[i]
                trades[-1]['exit_price'] = trade_price
                trades[-1]['pnl'] = net_pnl

                position_side = 0
                position_size = 0
                entry_price = 0

            else:
                # Check stop-loss
                stop_price = entry_price * (1 - stop_loss_frac) if position_side > 0 else entry_price * (1 + stop_loss_frac)
                if position_side > 0 and next_open < stop_price:
                    trade_price = next_open * (1 - slippage_rate)
                    pnl_points = trade_price - entry_price
                    exit_fee = trade_price * fee_rate * abs(position_size)
                    net_pnl = (pnl_points * position_size) - exit_fee
                    equity += net_pnl

                    trades[-1]['exit_time'] = df.index[i]
                    trades[-1]['exit_price'] = trade_price
                    trades[-1]['pnl'] = net_pnl

                    position_side = 0
                    position_size = 0
                    entry_price = 0

                elif position_side < 0 and next_open > stop_price:
                    trade_price = next_open * (1 + slippage_rate)
                    pnl_points = entry_price - trade_price
                    exit_fee = trade_price * fee_rate * abs(position_size)
                    net_pnl = (pnl_points * position_size) - exit_fee
                    equity += net_pnl

                    trades[-1]['exit_time'] = df.index[i]
                    trades[-1]['exit_price'] = trade_price
                    trades[-1]['pnl'] = net_pnl

                    position_side = 0
                    position_size = 0
                    entry_price = 0

        # If we are flat, consider entering
        if position_side == 0:
            # check cooldown
            if (i - last_entry_idx) >= cooldown_bars and equity > 0:
                # if signal is +1 or -1 => open new position
                if current_signal != 0:
                    fill_price = next_open * (1 + slippage_rate) if current_signal > 0 else next_open * (1 - slippage_rate)

                    # position_size => risk_in_$ = max_risk_percent*equity
                    # risk_in_$ = (fill_price*stop_loss_frac)*position_size
                    # => position_size = (max_risk_percent * equity)/(fill_price * stop_loss_frac)
                    position_size = (max_risk_percent * equity) / (fill_price * stop_loss_frac)
                    if position_size <= 0:
                        equity_curve.append(equity)
                        continue

                    # Pay entry fee
                    entry_fee = fill_price * fee_rate * abs(position_size)
                    if entry_fee >= equity:
                        equity_curve.append(equity)
                        continue

                    equity -= entry_fee

                    position_side = current_signal
                    entry_price = fill_price
                    position_entry_idx = i
                    last_entry_idx = i

                    trades.append({
                        'entry_time': df.index[i],
                        'entry_price': entry_price,
                        'side': 'LONG' if position_side > 0 else 'SHORT',
                        'exit_time': None,
                        'exit_price': None,
                        'pnl': None,
                        'position_size': position_size
                    })

        else:
            # if we do have a position and the signal flips to opposite/flat => exit
            if (position_side > 0 and current_signal <= 0) or (position_side < 0 and current_signal >= 0):
                trade_price = next_open
                if position_side > 0:
                    trade_price *= (1 - slippage_rate)
                    pnl_points = trade_price - entry_price
                else:
                    trade_price *= (1 + slippage_rate)
                    pnl_points = entry_price - trade_price

                exit_fee = trade_price * fee_rate * abs(position_size)
                net_pnl = (pnl_points * position_size) - exit_fee
                equity += net_pnl

                trades[-1]['exit_time'] = df.index[i]
                trades[-1]['exit_price'] = trade_price
                trades[-1]['pnl'] = net_pnl

                position_side = 0
                position_size = 0
                entry_price = 0

        if equity <= 0:
            # bankrupt
            equity_curve.append(equity)
            break

        # Record equity for this bar
        equity_curve.append(equity)

    total_pnl = equity - initial_capital
    pct_return = (total_pnl / initial_capital) * 100.0

    # Convert equity_curve to pd.Series
    # We'll have len(df)* or len(df)*2 items, but let's align it:
    # Easiest is to match them to df index if we appended once per bar iteration
    equity_series = pd.Series(equity_curve, index=range(len(equity_curve)))

    return trades, total_pnl, pct_return, equity_series

#############################################################################
# 3a) Compute Additional Stats: Sharpe, Sortino, Max Drawdown, etc.
#############################################################################
def compute_equity_curve_and_stats(equity_series, initial_capital=10000.0, bars_per_year=252):
    """
    Given an equity curve (pd.Series), compute:
      - Final net return
      - Bar-to-bar returns
      - Annualized Sharpe
      - Sortino
      - Max Drawdown
    We assume each bar ~ 1 day => bars_per_year = 252 for daily.
    For intraday, you might adjust bars_per_year accordingly.
    """
    # 1. Bar-to-bar returns
    # daily_ret[t] = (equity[t] - equity[t-1]) / equity[t-1]
    equity_values = equity_series.values
    daily_returns = np.diff(equity_values) / equity_values[:-1]

    if len(daily_returns) == 0:
        return {
            'sharpe': np.nan,
            'sortino': np.nan,
            'max_drawdown': np.nan,
            'annual_return': np.nan
        }

    # 2. Annualized Return
    total_ret = (equity_values[-1] / equity_values[0]) - 1.0
    # approximate annual return = (1 + total_ret)^(bars_per_year / len(daily_returns)) - 1
    ann_return = (1 + total_ret) ** (bars_per_year / len(daily_returns)) - 1

    # 3. Sharpe Ratio (assuming 0% risk-free)
    avg_daily = daily_returns.mean()
    std_daily = daily_returns.std(ddof=1)  # sample std
    if std_daily > 0:
        # annualization factor ~ sqrt(252)
        sharpe = (avg_daily / std_daily) * np.sqrt(bars_per_year)
    else:
        sharpe = np.nan

    # 4. Sortino Ratio
    # only consider negative returns in the denominator
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0:
        std_down = downside.std(ddof=1)
        sortino = (avg_daily / std_down) * np.sqrt(bars_per_year)
    else:
        sortino = np.nan

    # 5. Max Drawdown
    # typical approach: track the running max of the equity curve, see max percentage drop
    running_max = np.maximum.accumulate(equity_values)
    drawdown = (equity_values - running_max) / running_max
    max_dd = drawdown.min()  # negative number, e.g. -0.3 => 30% drawdown
    # we might store it as a positive percent:
    max_drawdown = abs(max_dd)

    stats = {
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_drawdown,
        'annual_return': ann_return
    }
    return stats

#############################################################################
# 3b) Analyze trades + Extended Stats
#############################################################################
def analyze_trades(trades, equity_series, initial_capital=10000.0):
    df_trades = pd.DataFrame(trades)
    if df_trades.empty:
        print("No trades were made.")
        return

    total_pnl = df_trades['pnl'].sum()
    pct_return = (total_pnl / initial_capital) * 100.0

    winners = df_trades[df_trades['pnl'] > 0]
    losers = df_trades[df_trades['pnl'] <= 0]
    win_rate = len(winners) / len(df_trades) if len(df_trades) > 0 else 0

    print("===== Backtest Results =====")
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total PnL: {total_pnl:.2f}")
    print(f"Pct Return: {pct_return:.2f}%")
    if not winners.empty:
        print(f"Average Win: {winners['pnl'].mean():.2f}")
    if not losers.empty:
        print(f"Average Loss: {losers['pnl'].mean():.2f}")
    print("Sample trades:")
    print(df_trades.tail(10))
    print("Worst 5 trades by PnL:")
    print(df_trades.nsmallest(5, 'pnl'))

    # Now compute additional stats from equity curve
    stats = compute_equity_curve_and_stats(equity_series, initial_capital=initial_capital, bars_per_year=252)
    print("\n===== Risk-Adjusted & Additional Metrics =====")
    print(f"Annualized Return: {stats['annual_return']*100:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe']:.3f}")
    print(f"Sortino Ratio: {stats['sortino']:.3f}")
    print(f"Max Drawdown: {stats['max_drawdown']*100:.2f}%")

#############################################################################
# 4) Main
#############################################################################
if __name__ == "__main__":
    csv_file = "historical_BTC.csv"
    df = pd.read_csv(csv_file)

    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # Example: tune indicators
    df = compute_indicators_and_signal(
        df,
        rsi_period=10,
        macd_fast=8,
        macd_slow=20,
        macd_signal=6,
        bb_period=18,
        bb_devup=2.0,
        bb_devdn=2.0
    )

    trades, total_pnl, pct_return, equity_series = run_backtest_on_df(
        df,
        initial_capital=10000.0,
        fee_rate=0.0005,
        slippage_rate=0.0002,
        max_risk_percent=0.02,
        stop_loss_frac=0.02,
        cooldown_bars=10,
        max_bars_in_trade=50
    )

    analyze_trades(trades, equity_series, initial_capital=10000.0)
