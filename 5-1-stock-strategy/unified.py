import pandas as pd
import numpy as np
import talib
import pytz
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional

class UnifiedStrategy:
    """
    This class handles both bull and bear strategies.
    Bull strategy: Long entries on oversold conditions (mean reversion)
    Bear strategy: Short entries on overbought conditions (momentum)
    """

    def __init__(self, config: Dict):
        """Initialize with config"""
        # ----- Strategy config -----
        self.config = config
        self.is_bear_config = "entry_mfi" in config  # Check if using bear config
        
        if self.is_bear_config:
            # Bear strategy parameters
            self.entry_mfi = config.get("entry_mfi", 70)  # Overbought for shorts
            self.exit_mfi = config.get("exit_mfi", 30)    # Oversold for short exits
            self.position_scale = config.get("position_scale", 3.5)
            self.stop_mult = config.get("stop_mult", 2.5)
            self.reward_risk = config.get("reward_risk", 1.5)
            self.trailing_stop = config.get("trailing_stop", True)
        else:
            # Original bull strategy parameters
            self.mfi_entry = config.get("mfi_entry", 35)
            self.bear_exit = config.get("bear_exit", 55)
            self.bull_exit = config.get("bull_exit", 75)
            self.position_scale = config.get("bull_position_scale", 3.5)
            self.stop_mult = config.get("stop_mult", 3.0)
            self.reward_risk = config.get("reward_risk", 1.5)
            self.trailing_stop = config.get("bull_trailing_stop", True)
        
        # Common parameters
        self.risk_per_trade = config.get("Risk Per Trade", 0.0075)
        self.min_stop_dollars = config.get("Min Stop Dollars", 1.00)
        self.max_stop_dollars = config.get("Max Stop Dollars", 2.50)
        self.max_hold_hours = config.get("Max Hold Hours", 5)
        self.min_hold_hours = config.get("Min Hold Hours", 0)
        self.market_slippage = config.get("market_slippage", 0.02)
        
        # Portfolio config
        self.initial_capital = config.get("Initial Capital", 100000)
        self.max_positions = config.get("Max Positions", 3)
        self.max_allocation = config.get("Max Allocation", 0.30)
        self.max_portfolio_risk = config.get("Max Portfolio Risk", 0.04)
        
        # Internal State
        self.capital = self.initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        self.symbol_cooldowns: Dict[str, datetime] = {}
        self.cooldown_minutes = 15

    def update_portfolio_value(self, current_prices: Dict[str,float]) -> float:
        """Calculate portfolio value accounting for both long and short positions"""
        value = self.capital
        for sym, pos in self.positions.items():
            if sym in current_prices:
                if pos.get('is_short', False):
                    # For shorts: profit/loss is inverse of price movement
                    value += pos['size'] * (pos['entry_price'] - current_prices[sym])
                else:
                    # For longs: profit/loss moves with price
                    value += pos['size'] * current_prices[sym]
        return value

    def _exit_position(self, symbol: str, price: float, timestamp: datetime, reason: str):
        """Close an existing position with a market order."""
        pos = self.positions[symbol]
        is_short = pos.get("is_short", False)
        
        # Calculate PnL based on position type
        if is_short:
            pnl = (pos["entry_price"] - price) * pos["size"]  # Short PnL
            action = "BUY"  # Buy to cover
        else:
            pnl = (price - pos["entry_price"]) * pos["size"]  # Long PnL
            action = "SELL"  # Sell to close
        
        # Update capital
        self.capital += price * pos["size"]
        
        # Record trade
        self.record_trade(
            timestamp=timestamp,
            symbol=symbol,
            action=action,
            price=price,
            size=pos["size"],
            reason=reason,
            pnl=pnl
        )
        # Remove the position
        del self.positions[symbol]

    def process_bar(self, timestamp: datetime, symbol_data: Dict[str,pd.Series]):
        """Process exits and entries based on strategy type"""
        # Build current_prices
        current_prices = {sym: bar["close"] for sym, bar in symbol_data.items()}

        # --- A) Exits first ---
        for sym in list(self.positions.keys()):
            if sym not in symbol_data:
                continue
            
            pos = self.positions[sym]
            bar = symbol_data[sym]
            price = bar["close"]
            is_short = pos.get("is_short", False)
            
            # 1) trailing stop update
            if self.trailing_stop:
                if is_short:
                    # For shorts: track lowest price and move stop down
                    pos["lowest_price"] = min(pos.get("lowest_price", pos["entry_price"]), bar["low"])
                    trail_dist = (pos["entry_price"] - pos["take_profit"]) * 0.5
                    new_stop = pos["lowest_price"] + trail_dist
                    pos["stop_loss"] = min(pos["stop_loss"], new_stop)  # Move stop down
                else:
                    # For longs: track highest price and move stop up
                    pos["highest_price"] = max(pos.get("highest_price", pos["entry_price"]), bar["high"])
                    trail_dist = (pos["take_profit"] - pos["entry_price"]) * 0.5
                    new_stop = pos["highest_price"] - trail_dist
                    pos["stop_loss"] = max(pos["stop_loss"], new_stop)  # Move stop up

            # 2) time-based exit
            hours_held = (timestamp - pos["entry_time"]).total_seconds() / 3600
            if hours_held > self.max_hold_hours:
                self._exit_position(sym, price, timestamp, "max_hold_time")
                continue
            
            # 3) check stop/target
            if is_short:
                hit_stop = (price >= pos["stop_loss"])
                hit_target = (price <= pos["take_profit"])
            else:
                hit_stop = (price <= pos["stop_loss"])
                hit_target = (price >= pos["take_profit"])
            
            if hit_stop or hit_target:
                if is_short:
                    exit_price = (pos["stop_loss"] + self.market_slippage) if hit_stop else (pos["take_profit"] + self.market_slippage)
                    pnl = (pos["entry_price"] - exit_price) * pos["size"]
                else:
                    exit_price = (pos["stop_loss"] - self.market_slippage) if hit_stop else (pos["take_profit"] - self.market_slippage)
                    pnl = (exit_price - pos["entry_price"]) * pos["size"]
                
                self.capital += exit_price * pos["size"]
                reason = "stop_or_target"
                
                # Record trade
                action = "BUY" if is_short else "SELL"
                self.record_trade(
                    timestamp=timestamp,
                    symbol=sym,
                    action=action,
                    price=exit_price,
                    size=pos["size"],
                    reason=reason,
                    pnl=pnl,
                    bar=bar
                )
                if hit_stop:
                    self.symbol_cooldowns[sym] = timestamp
                del self.positions[sym]
                continue

            if hours_held < self.min_hold_hours:
                continue
            
            # Technical exit conditions
            if is_short:
                # For shorts in bear strategy
                price_signal = bar["close"] > bar["vwap"]  # Price moves above VWAP
                mfi_signal = bar["mfi"] < self.exit_mfi    # MFI becomes oversold
            else:
                # For longs in bull strategy
                price_signal = bar["close"] > bar["vwap"]
                mfi_signal = bar["mfi"] > (self.bull_exit + 5)

            # Combine conditions
            reversal_signal = (price_signal or mfi_signal)

            # We want 2 consecutive bars that show 'reversal_signal'
            if reversal_signal:
                pos["reversal_count"] = pos.get("reversal_count", 0) + 1
            else:
                pos["reversal_count"] = 0

            # If we have 2 bars in a row with a reversal, exit
            if pos["reversal_count"] >= 2:
                if is_short:
                    exit_price = price + self.market_slippage
                    pnl = (pos["entry_price"] - exit_price) * pos["size"]
                else:
                    exit_price = price - self.market_slippage
                    pnl = (exit_price - pos["entry_price"]) * pos["size"]
                
                self.capital += exit_price * pos["size"]
                
                # Record trade
                action = "BUY" if is_short else "SELL"
                self.record_trade(
                    timestamp=timestamp,
                    symbol=sym,
                    action=action,
                    price=exit_price,
                    size=pos["size"],
                    reason="technical",
                    pnl=pnl,
                    bar=bar
                )
                del self.positions[sym]
                continue

        # --- B) New entries ---
        for sym, bar in symbol_data.items():
            if sym in self.positions:
                continue

            if self.check_cooldown(sym, timestamp):
                continue
            if len(self.positions) >= self.max_positions:
                continue

            can_trade, vol_mult = self.check_time_window(timestamp)
            if not can_trade:
                continue

            # Check regime
            regime = bar["regime"]
            if pd.isna(regime):
                continue
                
            # Filter regime based on strategy type
            if self.is_bear_config:
                if regime != "bear_high_vol":
                    continue
            else:
                if regime != "bull_high_vol":
                    continue

            # Entry conditions
            if self.is_bear_config:
                # Bear strategy: Short entry
                price_condition = bar["close"] < bar["vwap"]  # Price breaks below VWAP
                obv_condition = bar["obv_diff"] < 0          # OBV falling
                mfi_condition = bar["mfi"] > self.entry_mfi  # MFI overbought
            else:
                # Bull strategy: Long entry
                price_condition = bar["close"] > bar["vwap"]  # Price above VWAP
                obv_condition = bar["obv_diff"] > 0          # OBV rising
                mfi_condition = bar["mfi"] < self.mfi_entry  # MFI oversold

            if price_condition and obv_condition and mfi_condition:
                # Portfolio-based sizing
                price = bar["close"]
                atr_val = bar["atr"]
                portfolio_val = self.update_portfolio_value({sym: price})
                
                max_pos_value = portfolio_val * self.max_allocation
                risk_amount = portfolio_val * self.max_portfolio_risk
                
                raw_stop = self.stop_mult * atr_val
                stop_dist = min(max(raw_stop, self.min_stop_dollars), self.max_stop_dollars)
                
                risk_per_share = stop_dist
                base_size_risk = risk_amount / risk_per_share
                base_size_alloc = max_pos_value / price
                base_size = min(base_size_risk, base_size_alloc)
                size = int(round(base_size * self.position_scale * vol_mult))
                
                if size < 1:
                    continue

                cost = size * price
                if cost > self.capital:
                    continue

                if self.is_bear_config:
                    # Short position: stop above entry, target below
                    stop_loss = price + stop_dist
                    take_profit = price - (stop_dist * self.reward_risk)
                    action = "SELL"  # Short entry
                else:
                    # Long position: stop below entry, target above
                    stop_loss = price - stop_dist
                    take_profit = price + (stop_dist * self.reward_risk)
                    action = "BUY"  # Long entry

                self.positions[sym] = {
                    "entry_time": timestamp,
                    "entry_price": price,
                    "size": size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "regime": regime,
                    "highest_price": price if not self.is_bear_config else None,
                    "lowest_price": price if self.is_bear_config else None,
                    "is_short": self.is_bear_config
                }
                
                self.capital -= cost

                # Record trade
                self.record_trade(
                    timestamp=timestamp,
                    symbol=sym,
                    action=action,
                    price=price,
                    size=size,
                    reason="entry",
                    bar=bar
                )

    # Keep existing methods unchanged
    def check_cooldown(self, symbol: str, ts: datetime) -> bool:
        if symbol not in self.symbol_cooldowns:
            return False
        last_stop = self.symbol_cooldowns[symbol]
        minutes_since_stop = (ts - last_stop).total_seconds() / 60
        if minutes_since_stop < self.cooldown_minutes:
            return True
        else:
            del self.symbol_cooldowns[symbol]
            return False

    def check_time_window(self, ts: datetime):
        est = pytz.timezone('US/Eastern')
        if ts.tzinfo is None:
            ts = pytz.utc.localize(ts)
        local_time = ts.astimezone(est).time()

        if local_time < time(9,35) or local_time >= time(16,0):
            return (False, 0.0)
        if local_time < time(10,30):
            return (True, 0.8)
        elif local_time < time(12,0):
            return (True, 1.2)
        else:
            return (True, 1.0)

    def record_trade(
        self,
        timestamp: datetime,
        symbol: str,
        action: str,
        price: float,
        size: int,
        reason: str,
        pnl: float = None,
        bar: Optional[pd.Series] = None,
    ):
        current_prices = {symbol: price}
        portfolio_value = self.update_portfolio_value(current_prices)

        bar_high   = bar["high"]   if bar is not None and "high"   in bar else 0.0
        bar_low    = bar["low"]    if bar is not None and "low"    in bar else 0.0
        bar_close  = bar["close"]  if bar is not None and "close"  in bar else 0.0
        bar_regime = bar["regime"] if bar is not None and "regime" in bar else None

        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "price": price,
            "size": size,
            "regime": bar_regime,
            "reason": reason,
            "pnl": pnl if pnl else 0.0,
            "portfolio_value": portfolio_value,
            "capital": self.capital,
            "positions": len(self.positions),
            "bar_high":  float(bar_high),
            "bar_low":   float(bar_low),
            "bar_close": float(bar_close),
        }
        self.trades.append(trade)
        self.portfolio_history.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value,
            "capital": self.capital,
            "positions": len(self.positions)
        })

    def precompute_all_indicators(
        self,
        df: pd.DataFrame,
        regime_window: int = 20,
        volatility_percentile: float = 67,
        vwap_length: int = 50,
        mfi_length: int = 9
    ) -> pd.DataFrame:
        df = df.copy()
        required = ["open","high","low","close","volume"]
        if not all(c in df.columns for c in required):
            raise ValueError(f"Missing columns: {required}")

        df_shifted = df.shift(1)
        
        df_5min = df_shifted.resample('5min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        typical_price = (df_5min['high'] + df_5min['low'] + df_5min['close']) / 3
        roll_num = (typical_price * df_5min['volume']).rolling(vwap_length).mean()
        roll_den = df_5min['volume'].rolling(vwap_length).mean()
        df_5min['vwap'] = roll_num / roll_den.replace(0, np.nan)

        df_5min['obv'] = talib.OBV(df_5min['close'], df_5min['volume'])
        df_5min['obv_diff'] = df_5min['obv'].diff()

        df_5min['mfi'] = talib.MFI(
            df_5min['high'],
            df_5min['low'],
            df_5min['close'],
            df_5min['volume'],
            timeperiod=mfi_length
        )

        df_5min['atr'] = talib.ATR(
            df_5min['high'],
            df_5min['low'],
            df_5min['close'],
            timeperiod=2
        )

        df_5min['returns'] = df_5min['close'].pct_change()
        df_5min['smoothed_returns'] = df_5min['returns'].rolling(regime_window).mean() * 252
        df_5min['volatility'] = df_5min['returns'].rolling(regime_window).std() * np.sqrt(252)
        df_5min['vol_threshold'] = (
            df_5min['volatility']
                .rolling(regime_window)
                .quantile(volatility_percentile/100.0)
        )

        indicator_cols = [
            'vwap','obv','obv_diff','mfi','atr',
            'returns','smoothed_returns','volatility','vol_threshold'
        ]
        for col in indicator_cols:
            df[col] = df_5min[col].reindex(df.index).ffill()

        return df

    def fill_regime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        high_vol = df['volatility'] > df['vol_threshold']
        bull_mask = (df['smoothed_returns'] > 0) & high_vol
        bear_mask = (df['smoothed_returns'] <= 0) & high_vol

        df['regime'] = None
        df.loc[bull_mask, 'regime'] = 'bull_high_vol'
        df.loc[bear_mask, 'regime'] = 'bear_high_vol'
        return df

    def run(self, data: Dict[str, pd.DataFrame], config: Dict) -> List[Dict]:
        self.capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.portfolio_history.clear()
        self.symbol_cooldowns.clear()

        regime_window = config.get("Regime Window", 20)
        vol_perc = config.get("Volatility Percentile", 67)
        vwap_len = config.get("VWAP Length", 50)
        mfi_len = config.get("MFI Length", 9)

        processed_data = {}
        for sym, df in data.items():
            df_pre = self.precompute_all_indicators(
                df,
                regime_window=regime_window,
                volatility_percentile=vol_perc,
                vwap_length=vwap_len,
                mfi_length=mfi_len
            )
            df_reg = self.fill_regime_column(df_pre)
            processed_data[sym] = df_reg

        common_index = None
        for df in processed_data.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)

        if common_index is not None:
            common_index = common_index.sort_values()
            for ts in common_index:
                symbol_bar_data = {}
                for sym, df in processed_data.items():
                    if ts in df.index:
                        symbol_bar_data[sym] = df.loc[ts]
                self.process_bar(ts, symbol_bar_data)

        return self.trades

def run_unified_strategy(data: Dict[str, pd.DataFrame], config: Dict) -> List[Dict]:
    strat = UnifiedStrategy(config)
    trades = strat.run(data, config)
    return trades
