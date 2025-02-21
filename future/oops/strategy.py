import pandas as pd
import numpy as np
from typing import Dict, Tuple

class RefinedFuturesStrategy:
    def __init__(
        self,
        lookback_fast: int = 125,
        lookback_slow: int = 200,
        vol_lookback: int = 20,
        vol_target: float = 0.2,
        stop_atr_multiple: float = 1.5,
        partial_exit_1: float = 0.75,    # smaller threshold so partials can actually trigger
        partial_exit_2: float = 1.5,
        time_stop: int = 10,            # shorter time stop for demonstration
        trailing_stop_factor: float = 3.0,
        adx_threshold: float = None,
        scaling_mode: str = "none",     # "pyramid" to scale up
        corr_filter: float = False,     # if numeric, e.g. 0.8
        debug: bool = False,
    ):
        # Core MA parameters
        self.lookback_fast = lookback_fast
        self.lookback_slow = lookback_slow
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target
        self.stop_atr_multiple = stop_atr_multiple
        
        # Trading strategy parameters
        self.partial_exit_1 = partial_exit_1
        self.partial_exit_2 = partial_exit_2
        self.time_stop = time_stop
        self.trailing_stop_factor = trailing_stop_factor
        self.adx_threshold = adx_threshold
        self.scaling_mode = scaling_mode
        self.corr_filter = corr_filter  # correlation threshold
        self.debug = debug

        # Real-time portfolio state
        self.open_positions = {}  # {symbol: {'pos': float, 'trade_id': int, ...}}
        self.global_trade_id = 0  # increments with each new position

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate signals based on dual moving average and track validity."""
        df = df.copy()
        df["returns"] = df["Close"].pct_change(fill_method=None).fillna(0)

        # MAs
        df["ma_fast"] = df["Close"].rolling(self.lookback_fast).mean()
        df["ma_slow"] = df["Close"].rolling(self.lookback_slow).mean()

        df["signal_fast"] = np.where(df["Close"] > df["ma_fast"], 1, -1)
        df["signal_slow"] = np.where(df["Close"] > df["ma_slow"], 1, -1)

        df["signal"] = (df["signal_fast"] + df["signal_slow"]) / 2
        df["signal_valid"] = False
        valid_start = max(self.lookback_slow + 20, self.lookback_fast + 5)
        if valid_start < len(df):
            df.loc[df.index[valid_start:], "signal_valid"] = True

        return df

    def calculate_position_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility targeting to compute base position size."""
        df = df.copy()
        df["volatility"] = df["returns"].rolling(window=self.vol_lookback).std() * np.sqrt(252)
        df["position_size"] = self.vol_target / (df["volatility"] * np.sqrt(252))
        df["position"] = df["position_size"] * df["signal"]
        # Clip final position to ±1
        df["position"] = df["position"].clip(-1, 1)
        return df

    def day_by_day_correlation_filter(
        self, date, symbol, df_dict, daily_positions
    ) -> bool:
        """
        Example day-by-day correlation check:
          - 'daily_positions' is a dict {symbol: pos_value}
            indicating which instruments we hold *today*.
          - If 'corr_filter' is numeric, we skip adding a new symbol if
            correlation with any currently-held symbol > corr_filter
            using last 252 daily returns overlap.
        """
        if not self.corr_filter or self.corr_filter is False:
            return True  # No correlation filter

        # If no positions yet, always pass
        if not daily_positions:
            return True

        # Suppose 'df_dict' is a dict of {symbol: full DataFrame}
        # We find correlation of current symbol's recent returns with each open symbol's returns
        if symbol not in df_dict:
            return True  # no data?

        symbol_returns = df_dict[symbol]["Close"].pct_change(fill_method=None).fillna(0)
        # only up to 'date'
        symbol_returns = symbol_returns.loc[:date].tail(252)

        for s, pos in daily_positions.items():
            if s == symbol or pos == 0:
                continue
            # get last 252 returns for s as well
            s_returns = df_dict[s]["Close"].pct_change(fill_method=None).fillna(0)
            s_returns = s_returns.loc[:date].tail(252)
            if len(symbol_returns) < 10 or len(s_returns) < 10:
                continue
            corr = symbol_returns.corr(s_returns)
            if abs(corr) > self.corr_filter:
                if self.debug:
                    print(
                        f"Skipping {symbol} on {date.date()} due to correlation with {s} = {corr:.2f} > {self.corr_filter}"
                    )
                return False
        return True

    def apply_risk_management(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply partial exits, time stops, trailing stops, base stop, etc."""
        df = df.copy()
        # ATR
        df["tr"] = np.maximum(
            df["High"] - df["Low"],
            np.maximum(
                abs(df["High"] - df["Close"].shift(1)),
                abs(df["Low"] - df["Close"].shift(1))
            )
        )
        df["atr"] = df["tr"].rolling(window=self.vol_lookback).mean()

        # Initialize columns for risk mgmt
        df["position_adj"] = df["position"].copy()
        df["prev_pos"] = df["position"].shift(1).fillna(0)
        df["days_in_pos"] = 0.0
        df["exit1_hit"] = False
        df["exit2_hit"] = False
        df["size_multiplier"] = 1.0

        # We'll do a simple day-by-day loop here for clarity
        # so partial/time/trailing stops can be updated properly.
        # This is less "vectorized" but more transparent.
        open_trade = False
        exit1_hit = False
        exit2_hit = False
        days_in_pos = 0

        for i in range(1, len(df)):
            today = df.index[i]
            yday = df.index[i - 1]

            # Check if we have a position from yesterday
            prev_pos = df.at[yday, "position_adj"]
            pos_sign = np.sign(prev_pos)

            # If we had no position or sign changed, reset states
            if pos_sign == 0 or np.sign(df.at[today, "position"]) != pos_sign:
                open_trade = (pos_sign != 0)
                exit1_hit = False
                exit2_hit = False
                days_in_pos = 0
                df.at[today, "position_adj"] = df.at[today, "position"]
                continue

            # If same sign as yday, increment days_in_pos
            days_in_pos += 1

            # 1) time_stop
            if self.time_stop > 0 and days_in_pos >= self.time_stop:
                df.at[today, "position_adj"] = 0
                continue

            # 2) partial_exit checks
            if self.partial_exit_1 > 0 and not exit1_hit and pos_sign != 0:
                # For a long position
                if pos_sign > 0:
                    # partial_exit_1 => if today's Low < (ydayClose - partial_exit_1 * ydayATR)
                    yday_close = df.at[yday, "Close"]
                    yday_atr = df.at[yday, "atr"]
                    if df.at[today, "Low"] < (yday_close - self.partial_exit_1 * yday_atr):
                        df.at[today, "position_adj"] = prev_pos * 0.75
                        exit1_hit = True
                else:
                    # short side
                    yday_close = df.at[yday, "Close"]
                    yday_atr = df.at[yday, "atr"]
                    if df.at[today, "High"] > (yday_close + self.partial_exit_1 * yday_atr):
                        df.at[today, "position_adj"] = prev_pos * 0.75
                        exit1_hit = True

            if self.partial_exit_2 > 0 and not exit2_hit and pos_sign != 0:
                if pos_sign > 0:
                    # long side
                    yday_close = df.at[yday, "Close"]
                    yday_atr = df.at[yday, "atr"]
                    if df.at[today, "Low"] < (yday_close - self.partial_exit_2 * yday_atr):
                        df.at[today, "position_adj"] = (df.at[today, "position_adj"] * 0.5)
                        exit2_hit = True
                else:
                    # short side
                    yday_close = df.at[yday, "Close"]
                    yday_atr = df.at[yday, "atr"]
                    if df.at[today, "High"] > (yday_close + self.partial_exit_2 * yday_atr):
                        df.at[today, "position_adj"] = (df.at[today, "position_adj"] * 0.5)
                        exit2_hit = True

            # 3) trailing stop
            # if trailing_stop_factor > stop_atr_multiple, then do a bigger trailing approach
            if self.trailing_stop_factor > self.stop_atr_multiple and pos_sign != 0:
                # We'll track running high/low in a vector or an external approach, but for simplicity:
                # say we check "lowest close since entry" or "highest close since entry"
                # For brevity, let's do a short version:
                pass
                # (Implementation omitted for clarity)

            # 4) base_stop
            yday_atr = df.at[yday, "atr"]
            if pos_sign > 0:
                # if today's Low < ydayClose - stop_atr_multiple * ydayATR => exit
                if df.at[today, "Low"] < (df.at[yday, "Close"] - self.stop_atr_multiple * yday_atr):
                    df.at[today, "position_adj"] = 0
            else:
                # short side
                if df.at[today, "High"] > (df.at[yday, "Close"] + self.stop_atr_multiple * yday_atr):
                    df.at[today, "position_adj"] = 0

        df["position"] = df["position_adj"].clip(-1, 1)

        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["strat_returns"] = df["position"].shift(1).fillna(0) * df["returns"]
        df["cum_returns"] = (1 + df["strat_returns"]).cumprod()
        return df

    def run_single_instrument(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Run the logic on a single instrument's DataFrame."""
        df = self.calculate_signals(df)
        df = self.calculate_position_sizes(df)
        df = self.apply_risk_management(df)
        df = self.calculate_returns(df)

        std = df["strat_returns"].std()
        metrics = {
            "total_return": df["cum_returns"].iloc[-1] - 1,
            "annual_return": df["strat_returns"].mean() * 252,
            "annual_vol": std * np.sqrt(252),
            "sharpe_ratio": (
                df["strat_returns"].mean() * 252 / (std * np.sqrt(252))
                if std > 0 else 0
            ),
            "max_drawdown": (df["cum_returns"] / df["cum_returns"].cummax() - 1).min(),
            "win_rate": (df["strat_returns"] > 0).mean(),
        }
        return df, metrics

def analyze_portfolio(df_dict: Dict[str, pd.DataFrame], strategy: RefinedFuturesStrategy) -> pd.Series:
    """
    A day-by-day approach to handle correlation filtering dynamically.
    We'll build a final 'portfolio returns' series from daily PnL 
    of each instrument in a loop.
    """
    # Get common index
    common_index = None
    for sym, df in df_dict.items():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    common_index = pd.DatetimeIndex([pd.Timestamp(dt).tz_localize(None) for dt in common_index])
    common_index = common_index.sort_values()

    daily_returns = pd.DataFrame(index=common_index, columns=df_dict.keys(), dtype=float)

    # We'll keep track of each instrument's daily position, so we can build daily PnL
    daily_positions = {sym: pd.Series(0.0, index=common_index) for sym in df_dict.keys()}

    # Pre-run the single-instrument logic so we know each symbol's "desired" position
    # ignoring correlation. Then we'll do day-by-day correlation checks.
    single_results = {}
    for sym, df_raw in df_dict.items():
        df_aligned = df_raw.reindex(common_index).copy()
        # run single instrument logic
        df_out, _ = strategy.run_single_instrument(df_aligned)
        single_results[sym] = df_out

    # day-by-day correlation-based filtering + building portfolio returns
    for i, date in enumerate(common_index):
        # figure out which symbols are "on" at this date ignoring correlation
        # then see if we skip any due to correlation
        current_positions = {}  # symbol -> position
        for sym, df_out in single_results.items():
            if date in df_out.index:
                desired_pos = df_out.at[date, "position"]
                # if not valid or NaN, skip
                if np.isnan(desired_pos) or desired_pos == 0:
                    continue
                # correlation check
                if not strategy.day_by_day_correlation_filter(date, sym, df_dict, current_positions):
                    desired_pos = 0
                if desired_pos != 0:
                    current_positions[sym] = desired_pos

        # Now we have final positions for each symbol on this date
        for sym, pos in current_positions.items():
            daily_positions[sym].iat[i] = pos

    # Now compute daily returns for each symbol
    for sym in df_dict.keys():
        # daily PnL = position[t-1] * returns[t]
        # so let's do a shift(1) approach
        ret_series = df_dict[sym]["Close"].pct_change(fill_method=None).fillna(0).reindex(common_index)
        pos_series = daily_positions[sym].shift(1).fillna(0)
        daily_returns[sym] = pos_series * ret_series

    # sum across symbols, then scale final portfolio to strategy.vol_target
    daily_portfolio = daily_returns.mean(axis=1)  # equal weight by symbol
    # scale volatility
    realized_vol = daily_portfolio.std() * np.sqrt(252)
    scale_factor = strategy.vol_target / realized_vol if realized_vol > 0 else 1.0
    scaled_portfolio = daily_portfolio * scale_factor

    # Build final
    cum_returns = (1 + scaled_portfolio).cumprod()
    ann_ret = scaled_portfolio.mean() * 252
    ann_vol = scaled_portfolio.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = (cum_returns / cum_returns.cummax() - 1).min()

    return pd.Series({
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    })
