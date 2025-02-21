import pandas as pd
import numpy as np
import talib
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Union

# -----------------------------
# 1) Single-Pass Precomputation
# -----------------------------
def precompute_all_indicators(
    df: pd.DataFrame, 
    regime_window: int = 20,
    volatility_percentile: float = 67
) -> pd.DataFrame:
    """
    Create a copy of 'df' with columns:
        'vwap', 'obv_diff', 'mfi', 'atr'
        'returns', 'smoothed_returns', 'volatility', 'vol_67_pct'
    but we do NOT finalize 'regime' here, so that 
    we can apply different regime logic if needed.
    """
    # Defensive copy
    df = df.copy()
    
    # Basic checks
    if not all(c in df.columns for c in ["open","high","low","close","volume"]):
        raise ValueError("DataFrame must have columns: open, high, low, close, volume.")
    if len(df) < regime_window:
        # We'll let the code proceed, but obviously you'll have fewer valid bars
        pass

    # 1) VWAP
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    vwap_series = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
    df["vwap"] = vwap_series.values  # store as np array

    # 2) OBV + diff
    df["obv"] = talib.OBV(df["close"], df["volume"])
    df["obv_diff"] = df["obv"].diff()

    # 3) MFI
    #   If your MFI period is always 9, you can set that. 
    #   or pass it in if you need.
    df["mfi"] = talib.MFI(
        df["high"], 
        df["low"], 
        df["close"], 
        df["volume"], 
        timeperiod=9
    )

    # 4) ATR
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=2)

    # 5) Returns and Volatility calculations for regime
    df["returns"] = df["close"].pct_change()
    
    # Annualized smoothed returns (SMA * 252)
    df["smoothed_returns"] = df["returns"].rolling(window=regime_window, min_periods=1).mean() * 252
    
    # Annualized volatility (StdDev * sqrt(252))
    df["volatility"] = df["returns"].rolling(window=regime_window, min_periods=1).std() * np.sqrt(252)
    
    # Configurable percentile volatility threshold (rolling)
    df["vol_threshold"] = df["volatility"].rolling(window=regime_window, min_periods=1).quantile(volatility_percentile/100)

    return df


def fill_regime_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create or overwrite df["regime"] in one pass:
        bull_high_vol, bear_high_vol, or None
    based on:
        volatility > 67th percentile volatility
        direction => bull if smoothed_returns > 0 else bear
    """
    # Defensive copy so we don't mutate
    df = df.copy()
    required_cols = ["volatility", "vol_threshold", "smoothed_returns"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError(f"DataFrame must have columns: {', '.join(required_cols)}")
    
    # 1) High volatility mask using configurable threshold
    high_vol_mask = df["volatility"] > df["vol_threshold"]
    
    # 2) Direction based on smoothed returns
    bull_mask = (df["smoothed_returns"] > 0) & high_vol_mask
    bear_mask = (df["smoothed_returns"] <= 0) & high_vol_mask
    
    # 3) Fill regime
    df["regime"] = None
    df.loc[bull_mask, "regime"] = "bull_high_vol"
    df.loc[bear_mask, "regime"] = "bear_high_vol"

    return df

# -----------------------------------------------
# 2) The Strategy, referencing precomputed fields
# -----------------------------------------------
class VAMEStrategy:
    def __init__(self, config: Dict):
        """
        We assume config includes:
          MFI thresholds, 
          Min/Max stop dollars, 
          Risk per trade, 
          etc.
        """
        self.config = config
        self.mfi_entry   = config.get("mfi_entry", 30)
        self.bear_exit   = config.get("bear_exit", 55)
        self.bull_exit   = config.get("bull_exit", 75)
        self.risk_per_trade   = config.get("Risk Per Trade", 0.025)
        self.min_stop_dollars = config.get("Min Stop Dollars", 1.00)
        self.max_stop_dollars = config.get("Max Stop Dollars", 2.50)
        # For minute data, default to 2 hours max hold time
        self.max_hold_hours   = config.get("Max Hold Hours", 2)
        
        # Get position scales and trailing stops
        bear_position_scale = config.get("bear_position_scale", 2.25)
        bull_position_scale = config.get("bull_position_scale", 1.50)
        bear_trailing_stop = config.get("bear_trailing_stop", True)
        bull_trailing_stop = config.get("bull_trailing_stop", True)
        
        # Get base parameters
        base_reward_risk = config.get("reward_risk", 2.0)
        base_stop_mult = config.get("stop_mult", 1.6)
        
        # Regime-specific
        self.regime_params = {
            "bear_high_vol": {
                "position_scale": bear_position_scale,
                "reward_risk": base_reward_risk,
                "stop_mult": base_stop_mult,
                "mfi_overbought": self.bear_exit,
                "trailing_stop": bear_trailing_stop,
            },
            "bull_high_vol": {
                "position_scale": bull_position_scale,
                "reward_risk": base_reward_risk,
                "stop_mult": base_stop_mult,
                "mfi_overbought": self.bull_exit,
                "trailing_stop": bull_trailing_stop,
            },
        }
        
        self.position = None
        self.trades   = []
        self.highest_price = None

    def get_regime_parameters(self, regime: str) -> Dict:
        return self.regime_params.get(regime, {})

    def check_time_window(self, timestamp: datetime) -> Tuple[bool, float]:
        morning_start   = time(9, 30)
        midmorning      = time(10, 30)
        afternoon_start = time(12, 0)
        market_close    = time(16, 0)

        current_time = timestamp.time()
        if current_time < morning_start:
            return False, 0.0
        elif morning_start <= current_time < midmorning:
            return True, 0.8
        elif midmorning <= current_time < afternoon_start:
            return True, 1.2
        elif afternoon_start <= current_time < market_close:
            return True, 1.0
        else:
            return False, 0.0

    def calculate_position_size(
        self,
        price: float,
        atr_val: float,
        capital: float,
        regime_params: Dict,
        vol_mult: float
    ) -> int:
        risk_amount = capital * self.risk_per_trade
        risk_per_share = atr_val * regime_params["stop_mult"]
        base_size = risk_amount / risk_per_share
        position_size = round(base_size * regime_params["position_scale"] * vol_mult)
        return max(1, position_size)

    def run(self, df: pd.DataFrame) -> List[Dict]:
        self.trades = []
        self.position = None
        self.highest_price = None
        
        # For minute data, we just need enough bars for the regime window
        min_window = self.config.get("Regime Window", 20)  # typically 20 bars
        if len(df) < min_window:
            return []

        # Main loop
        for i in range(min_window, len(df)):
            current_bar = df.iloc[i]
            ts = current_bar.name  # typically a Timestamp index

            # Step A: Are we in a position?
            if not self.position:
                # Attempt entry
                tradeable, vol_mult = self.check_time_window(ts)
                if not tradeable:
                    continue
                # Check if regime is high vol
                regime = current_bar["regime"]
                if pd.isna(regime):
                    continue  # not high vol

                # Strategy conditions using close for signals
                signal_price = current_bar["close"]
                price_below_vwap = signal_price < current_bar["vwap"]
                obv_falling      = current_bar["obv_diff"] < 0
                mfi_oversold     = current_bar["mfi"] < self.mfi_entry

                if price_below_vwap and obv_falling and mfi_oversold:
                    # regime params
                    params = self.get_regime_parameters(regime)
                    if not params:
                        continue

                    # Set limit order slightly above close for pullback entry
                    limit_price = signal_price * 1.001  # 0.1% above close
                    atr_val = current_bar["atr"]
                    
                    # Position sizing using limit price
                    size = self.calculate_position_size(
                        limit_price, atr_val, 100000, params, vol_mult
                    )

                    # Stop loss / take profit as limit orders
                    raw_stop = params["stop_mult"] * atr_val
                    stop_dist = min(
                        max(raw_stop, self.min_stop_dollars),
                        self.max_stop_dollars
                    )
                    stop_loss   = limit_price - stop_dist
                    take_profit = limit_price + (stop_dist * params["reward_risk"])

                    # trailing stop as limit order
                    self.highest_price = limit_price if params.get("trailing_stop", False) else None
                    # record position with limit orders
                    self.position = {
                        "entry_time": ts,
                        "entry_price": limit_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "size": size,
                        "regime": regime,
                    }
                    self.trades.append({
                        "timestamp": ts,
                        "action": "BUY",
                        "price": limit_price,  # Use limit price for entry
                        "size": size,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "regime": regime
                    })
            else:
                # Step B: Check limit orders
                current_price = current_bar["close"]
                regime = self.position["regime"]
                params = self.get_regime_parameters(regime)
                
                # Update trailing stop limit order
                if params.get("trailing_stop", False) and self.highest_price is not None:
                    self.highest_price = max(self.highest_price, current_price)
                    trailing_stop = self.highest_price - (
                        self.position["take_profit"] - self.position["entry_price"]
                    ) * 0.5
                    # Update stop loss limit order
                    self.position["stop_loss"] = max(self.position["stop_loss"], trailing_stop)

                # Time-based exit at market
                hours_held = (ts - self.position["entry_time"]).total_seconds() / 3600
                if hours_held > self.max_hold_hours:
                    # Market order for time-based exit
                    pnl = (current_price - self.position["entry_price"]) * self.position["size"]
                    self.trades.append({
                        "timestamp": ts,
                        "action": "SELL",
                        "price": current_price,  # Market order for time-based exit
                        "size": self.position["size"],
                        "pnl": pnl,
                        "reason": "max_hold_time",
                        "regime": regime
                    })
                    self.position = None
                    self.highest_price = None
                    continue

                # Check if limit orders would be hit
                if current_price <= self.position["stop_loss"] or current_price >= self.position["take_profit"]:
                    # Use limit price for exit
                    exit_price = (self.position["stop_loss"] if current_price <= self.position["stop_loss"] 
                                else self.position["take_profit"])
                    pnl = (exit_price - self.position["entry_price"]) * self.position["size"]
                    self.trades.append({
                        "timestamp": ts,
                        "action": "SELL",
                        "price": exit_price,  # Use limit price for stop/target
                        "size": self.position["size"],
                        "pnl": pnl,
                        "reason": "stop_or_target",
                        "regime": regime
                    })
                    self.position = None
                    self.highest_price = None
                    continue

                # Technical exit with limit order
                price_above_vwap = current_price > current_bar["vwap"]
                mfi_overbought   = current_bar["mfi"] > params.get("mfi_overbought", 70)

                if price_above_vwap or mfi_overbought:
                    # Place limit order slightly below current price
                    exit_price = current_price * 0.999  # 0.1% below current price
                    pnl = (exit_price - self.position["entry_price"]) * self.position["size"]
                    self.trades.append({
                        "timestamp": ts,
                        "action": "SELL",
                        "price": exit_price,  # Use limit price for technical exit
                        "size": self.position["size"],
                        "pnl": pnl,
                        "reason": "technical",
                        "regime": regime
                    })
                    self.position = None
                    self.highest_price = None

        return self.trades

# ------------------------------
# 3) Example usage for a config
# ------------------------------
def run_single_config(df: pd.DataFrame, config: Dict) -> List[Dict]:
    """
    A convenient function that:
      1) precomputes everything in a single pass
      2) applies fill_regime_column with new regime logic
      3) runs the VAMEStrategy
    """
    # Get parameters
    regime_window = config.get("Regime Window", 20)
    volatility_percentile = config.get("Volatility Percentile", 67)

    # 1) Precompute
    df_pre = precompute_all_indicators(
        df,
        regime_window=regime_window,
        volatility_percentile=volatility_percentile
    )

    # 2) Fill in 'regime' column
    df_reg = fill_regime_column(df_pre)

    # 3) Initialize + run
    strat = VAMEStrategy(config)
    trades = strat.run(df_reg)
    return trades
