import pandas as pd
import numpy as np
import talib
import pytz
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Union


# -----------------------------
# 1) Single-Pass Precomputation
# -----------------------------
def precompute_all_indicators(
    df: pd.DataFrame, 
    regime_window: int = 20,
    volatility_percentile: float = 67,
    vwap_length: int = 50,    # match the 50 from Pinescript
    mfi_length: int = 9       # match the 9 from Pinescript
) -> pd.DataFrame:
    """
    Create a copy of 'df' with columns:
        'vwap_50', 'obv_diff', 'mfi', 'atr'
        'returns', 'smoothed_returns', 'volatility', 'vol_threshold'
    but we do NOT finalize 'regime' here, so that 
    we can apply different regime logic if needed.
    """
    # Defensive copy
    df = df.copy()
    
    # Basic checks
    required = ["open","high","low","close","volume"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"DataFrame must have columns: {required}.")
    if len(df) < regime_window:
        # We'll let the code proceed, but obviously you'll have fewer valid bars
        pass

    # 1) 50-bar rolling VWAP (matching PineScript: ta.sma(typical*volume,50) / ta.sma(volume,50))
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    # Rolling SMA of typical*volume over the last 50 bars:
    rolling_num = (typical_price * df["volume"]).rolling(vwap_length).mean()
    # Rolling SMA of volume over the last 50 bars:
    rolling_den = df["volume"].rolling(vwap_length).mean()
    # 50-bar rolling VWAP:
    df["vwap"] = rolling_num / (rolling_den.replace(0, np.nan))

    # 2) OBV + diff
    df["obv"] = talib.OBV(df["close"], df["volume"])
    df["obv_diff"] = df["obv"].diff()

    # 3) MFI (same logic as PineScript’s ta.mfi(hlc3, 9))
    #    Standard TA-Lib MFI uses (high,low,close,volume). 
    #    PineScript’s MFI(hlc3) is effectively the same standard MFI.
    df["mfi"] = talib.MFI(
        df["high"], 
        df["low"], 
        df["close"], 
        df["volume"], 
        timeperiod=mfi_length
    )

    # 4) ATR (the Pine code doesn't show ATR usage for signals, 
    #    but your strategy uses it for sizing — so we keep it.)
    df["atr"] = talib.ATR(df["high"], df["low"], df["close"], timeperiod=2)

    # 5) Returns and volatility calculations for regime logic
    df["returns"] = df["close"].pct_change()
    
    # Annualized smoothed returns (SMA * 252)
    df["smoothed_returns"] = (
        df["returns"]
        .rolling(window=regime_window, min_periods=regime_window)
        .mean() * 252
    )
    
    # Annualized volatility (StdDev * sqrt(252))
    df["volatility"] = (
        df["returns"]
        .rolling(window=regime_window, min_periods=regime_window)
        .std() 
        * np.sqrt(252)
    )
    
    # Rolling percentile threshold for volatility (similar to ta.percentile_nearest_rank())
    df["vol_threshold"] = (
        df["volatility"]
        .rolling(window=regime_window, min_periods=regime_window)
        .quantile(volatility_percentile / 100.0)
    )

    return df


def fill_regime_column(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create or overwrite df["regime"]:
        'bull_high_vol', 'bear_high_vol', or NaN
    based on:
      - volatility > rolling threshold
      - direction => bull if smoothed_returns > 0 else bear
    """
    df = df.copy()
    required_cols = ["volatility", "vol_threshold", "smoothed_returns"]
    if not all(c in df.columns for c in required_cols):
        raise ValueError(
            f"DataFrame must have columns: {', '.join(required_cols)}"
        )
    
    # 1) High volatility mask
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
        """Initialize strategy with config"""
        self.config = config
        self.mfi_entry = config.get("mfi_entry", 30)
        self.bear_exit = config.get("bear_exit", 55)
        self.bull_exit = config.get("bull_exit", 75)
        self.risk_per_trade = config.get("Risk Per Trade", 0.025)
        self.min_stop_dollars = config.get("Min Stop Dollars", 1.00)
        self.max_stop_dollars = config.get("Max Stop Dollars", 2.50)
        self.max_hold_hours = config.get("Max Hold Hours", 2)
        
        # Get position scales and trailing stops
        bear_position_scale = config.get("bear_position_scale", 2.25)
        bull_position_scale = config.get("bull_position_scale", 1.50)
        bear_trailing_stop = config.get("bear_trailing_stop", True)
        bull_trailing_stop = config.get("bull_trailing_stop", True)
        
        # Get base parameters
        base_reward_risk = config.get("reward_risk", 2.0)
        base_stop_mult = config.get("stop_mult", 1.6)
        
        # Regime-specific parameters
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
        
        # State variables
        self.position = None
        self.trades = []
        self.highest_price = None
        self.pending_limit_order = None
        self.last_stop_loss = None  # Track last stop loss time
        self.cooldown_minutes = 15  # Wait period after stop loss
        
    def get_regime_parameters(self, regime: str) -> Dict:
        return self.regime_params.get(regime, {})

    def check_time_window(self, timestamp: datetime) -> Tuple[bool, float]:
        """Check if we can trade and get volume multiplier"""
        market_open = time(9, 30)
        trading_start = time(9, 35)  # Start trading 5 minutes after open
        midmorning = time(10, 30)
        afternoon_start = time(12, 0)
        market_close = time(16, 0)

        # Convert timestamp to Eastern time for market hours check
        est = pytz.timezone('US/Eastern')
        if timestamp.tzinfo is None:
            timestamp = pytz.utc.localize(timestamp)
        current_time = timestamp.astimezone(est).time()
        if current_time < trading_start:  # Block first 5 minutes
            return False, 0.0
        elif trading_start <= current_time < midmorning:
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
        """Run strategy with limit orders based on previous bar's close"""
        self.trades = []
        self.position = None
        self.highest_price = None
        self.pending_limit_order = None
        
        min_window = self.config.get("Regime Window", 20)
        if len(df) < min_window:
            return []

        # Main loop
        for i in range(min_window, len(df)):
            current_bar = df.iloc[i]
            prev_bar = df.iloc[i-1]  # Previous bar for limit price
            ts = current_bar.name

            # Step A: Check pending limit order
            if self.pending_limit_order:
                # Check if limit order would be filled (price traded through limit)
                if current_bar["low"] <= self.pending_limit_order["limit_price"]:
                    # Fill at limit price
                    self.position = {
                        "entry_time": ts,
                        "entry_price": self.pending_limit_order["limit_price"],
                        "stop_loss": self.pending_limit_order["stop_loss"],
                        "take_profit": self.pending_limit_order["take_profit"],
                        "size": self.pending_limit_order["size"],
                        "regime": self.pending_limit_order["regime"]
                    }
                    
                    # Record trade
                    self.trades.append({
                        "timestamp": ts,
                        "action": "BUY",
                        "price": self.pending_limit_order["limit_price"],
                        "size": self.pending_limit_order["size"],
                        "stop_loss": self.pending_limit_order["stop_loss"],
                        "take_profit": self.pending_limit_order["take_profit"],
                        "regime": self.pending_limit_order["regime"]
                    })
                    
                    # Reset limit order
                    self.pending_limit_order = None
                    
                    # Set highest price for trailing stop
                    if self.position["regime"].get("trailing_stop", False):
                        self.highest_price = self.position["entry_price"]
                        
                    # Clear any cooldown since we're entering a new position
                    self.last_stop_loss = None
                        
                # Cancel limit if not filled after 2 bars
                elif ts - self.pending_limit_order["timestamp"] > pd.Timedelta(minutes=2):
                    self.pending_limit_order = None

            # Step B: Look for new entry if no position and no pending order
            elif not self.position and not self.pending_limit_order:
                tradeable, vol_mult = self.check_time_window(ts)
                if not tradeable:
                    continue

                regime = current_bar["regime"]
                if pd.isna(regime):
                    continue

                # Check cooldown period after stop loss
                if self.last_stop_loss is not None:
                    minutes_since_stop = (ts - self.last_stop_loss).total_seconds() / 60
                    if minutes_since_stop < self.cooldown_minutes:
                        continue

                # Check entry conditions
                price_below_vwap = current_bar["close"] < current_bar["vwap"]
                obv_falling = current_bar["obv_diff"] < 0
                mfi_oversold = current_bar["mfi"] < self.mfi_entry

                if price_below_vwap and obv_falling and mfi_oversold:
                    params = self.get_regime_parameters(regime)
                    if not params:
                        continue

                    # Place limit order at previous bar's close
                    limit_price = prev_bar["close"]
                    atr_val = current_bar["atr"]
                    
                    # Calculate position size
                    size = self.calculate_position_size(
                        limit_price, atr_val, 100000, params, vol_mult
                    )

                    # Calculate stops based on limit price
                    raw_stop = params["stop_mult"] * atr_val
                    stop_dist = min(
                        max(raw_stop, self.min_stop_dollars),
                        self.max_stop_dollars
                    )
                    stop_loss = limit_price - stop_dist
                    take_profit = limit_price + (stop_dist * params["reward_risk"])
                    
                    # Store pending limit order
                    self.pending_limit_order = {
                        "timestamp": ts,
                        "limit_price": limit_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "size": size,
                        "regime": regime
                    }

            # Step C: Check exits if in position
            elif self.position:
                price = current_bar["close"]
                regime = self.position["regime"]
                params = self.get_regime_parameters(regime)
                
                # Update trailing stop
                if params.get("trailing_stop", False) and self.highest_price is not None:
                    self.highest_price = max(self.highest_price, price)
                    trailing_stop = self.highest_price - (
                        self.position["take_profit"] - self.position["entry_price"]
                    ) * 0.5
                    self.position["stop_loss"] = max(
                        self.position["stop_loss"], trailing_stop
                    )

                # Time-based exit
                hours_held = (ts - self.position["entry_time"]).total_seconds() / 3600
                if hours_held > self.max_hold_hours:
                    pnl = (price - self.position["entry_price"]) * self.position["size"]
                    self.trades.append({
                        "timestamp": ts,
                        "action": "SELL",
                        "price": price,
                        "size": self.position["size"],
                        "pnl": pnl,
                        "reason": "max_hold_time",
                        "regime": regime
                    })
                    self.position = None
                    self.highest_price = None
                    continue

                # Check stops
                if price <= self.position["stop_loss"] or price >= self.position["take_profit"]:
                    # Use exact stop/target prices
                    hit_stop = price <= self.position["stop_loss"]
                    exit_price = (self.position["stop_loss"] if hit_stop
                                else self.position["take_profit"])
                    pnl = (exit_price - self.position["entry_price"]) * self.position["size"]
                    self.trades.append({
                        "timestamp": ts,
                        "action": "SELL",
                        "price": exit_price,
                        "size": self.position["size"],
                        "pnl": pnl,
                        "reason": "stop_or_target",
                        "regime": regime
                    })
                    
                    # Start cooldown timer if stopped out
                    if hit_stop:
                        self.last_stop_loss = ts
                        
                    self.position = None
                    self.highest_price = None
                    continue

                # Technical exit
                price_above_vwap = price > current_bar["vwap"]
                mfi_overbought = current_bar["mfi"] > params.get("mfi_overbought", 70)
                
                if price_above_vwap or mfi_overbought:
                    pnl = (price - self.position["entry_price"]) * self.position["size"]
                    self.trades.append({
                        "timestamp": ts,
                        "action": "SELL",
                        "price": price,
                        "size": self.position["size"],
                        "pnl": pnl,
                        "reason": "technical",
                        "regime": regime
                    })
                    self.position = None
                    self.highest_price = None

        return self.trades

def run_single_config(df: pd.DataFrame, config: Dict) -> List[Dict]:
    """Run strategy with given config"""
    # Get parameters
    regime_window = config.get("Regime Window", 20)
    volatility_percentile = config.get("Volatility Percentile", 67)
    vwap_length = config.get("VWAP Length", 50)
    mfi_length = config.get("MFI Length", 9)

    # 1) Precompute indicators
    df_pre = precompute_all_indicators(
        df,
        regime_window=regime_window,
        volatility_percentile=volatility_percentile,
        vwap_length=vwap_length,
        mfi_length=mfi_length
    )

    # 2) Fill regime column
    df_reg = fill_regime_column(df_pre)

    # 3) Run strategy
    strat = VAMEStrategy(config)
    trades = strat.run(df_reg)
    return trades
