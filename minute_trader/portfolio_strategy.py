from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from vectorized_strategy import VAMEStrategy, precompute_all_indicators, fill_regime_column

class PortfolioManager:
    def __init__(self, config: Dict):
        """Initialize portfolio manager"""
        self.initial_capital = config.get("Initial Capital", 100000)
        self.max_positions = config.get("Max Positions", 5)
        self.max_allocation = config.get("Max Allocation", 0.20)  # 20% max per position
        self.max_portfolio_risk = config.get("Max Portfolio Risk", 0.02)  # 2% portfolio risk
        self.config = config  # Store config for strategy initialization
        
        # Portfolio state
        self.capital = self.initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> position details
        self.trades: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        
        # Strategy instance for each symbol
        self.strategies: Dict[str, VAMEStrategy] = {}
        
        # Cooldown tracking per symbol
        self.symbol_cooldowns: Dict[str, datetime] = {}  # symbol -> last stop loss time
        self.cooldown_minutes = 15  # Wait period after stop loss
        
    def update_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value including cash and positions"""
        portfolio_value = self.capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = position["size"] * current_price
                portfolio_value += position_value
                
        return portfolio_value
        
    def check_cooldown(self, symbol: str, timestamp: datetime) -> bool:
        """Check if symbol is in cooldown period after stop loss"""
        if symbol in self.symbol_cooldowns:
            last_stop = self.symbol_cooldowns[symbol]
            minutes_since_stop = (timestamp - last_stop).total_seconds() / 60
            if minutes_since_stop < self.cooldown_minutes:
                return True  # Still in cooldown
            else:
                del self.symbol_cooldowns[symbol]  # Clear expired cooldown
        return False
        
    def calculate_position_size(self, 
        symbol: str,
        price: float,
        atr: float,
        regime_params: Dict,
        vol_mult: float
    ) -> int:
        """Calculate position size considering portfolio constraints"""
        portfolio_value = self.update_portfolio_value({symbol: price})
        
        if len(self.positions) >= self.max_positions:
            return 0
            
        max_position_value = portfolio_value * self.max_allocation
        risk_amount = portfolio_value * self.max_portfolio_risk
        risk_per_share = atr * regime_params["stop_mult"]
        risk_based_size = risk_amount / risk_per_share
        allocation_based_size = max_position_value / price
        
        base_size = min(risk_based_size, allocation_based_size)
        position_size = round(base_size * regime_params["position_scale"] * vol_mult)
        
        position_cost = position_size * price
        if position_cost > self.capital:
            return 0
            
        return max(1, position_size)
        
    def record_trade(self,
        timestamp: datetime,
        symbol: str,
        action: str,
        price: float,
        size: int,
        regime: str,
        reason: Optional[str] = None,
        pnl: Optional[float] = None
    ):
        """Record trade with portfolio context"""
        portfolio_value = self.update_portfolio_value({symbol: price})
        
        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action,
            "price": price,
            "size": size,
            "regime": regime,
            "reason": reason,
            "pnl": pnl,
            "portfolio_value": portfolio_value,
            "capital": self.capital,
            "positions": len(self.positions)
        }
        
        self.trades.append(trade)
        self.portfolio_history.append({
            "timestamp": timestamp,
            "portfolio_value": portfolio_value,
            "capital": self.capital,
            "positions": len(self.positions)
        })
        
    def process_bar(self, timestamp: datetime, symbol_data: Dict[str, pd.Series]):
        """Process new bar data for all symbols"""
        current_prices = {
            symbol: data["close"] 
            for symbol, data in symbol_data.items()
        }
        
        # First process exits
        for symbol in list(self.positions.keys()):
            if symbol not in symbol_data:
                continue
                
            position = self.positions[symbol]
            current_bar = symbol_data[symbol]
            strategy = self.strategies[symbol]
            
            # Check exit conditions
            price = current_bar["close"]
            regime = current_bar["regime"]
            params = strategy.get_regime_parameters(regime)
            
            # Update trailing stop if needed
            if params.get("trailing_stop", False):
                trailing_stop = position["highest_price"] - (
                    position["take_profit"] - position["entry_price"]
                ) * 0.5
                position["stop_loss"] = max(position["stop_loss"], trailing_stop)
            
            # Check stops
            hit_stop = price <= position["stop_loss"]
            hit_target = price >= position["take_profit"]
            if hit_stop or hit_target:
                # Start cooldown if stopped out
                if hit_stop:
                    self.symbol_cooldowns[symbol] = timestamp
                self._exit_position(symbol, price, timestamp, "stop_or_target")
                continue
                
            # Check technical exit
            price_above_vwap = price > current_bar["vwap"]
            mfi_overbought = current_bar["mfi"] > params.get("mfi_overbought", 70)
            if price_above_vwap or mfi_overbought:
                self._exit_position(symbol, price, timestamp, "technical")
                continue
                
            # Update highest price for trailing stop
            position["highest_price"] = max(position["highest_price"], price)
            
        # Then look for entries
        for symbol, current_bar in symbol_data.items():
            if symbol in self.positions:
                continue
                
            # Check cooldown period
            if self.check_cooldown(symbol, timestamp):
                continue
                
            strategy = self.strategies.get(symbol)
            if not strategy:
                strategy = VAMEStrategy(self.config)
                self.strategies[symbol] = strategy
                
            # Check entry conditions
            tradeable, vol_mult = strategy.check_time_window(timestamp)
            if not tradeable:
                continue
                
            regime = current_bar["regime"]
            if pd.isna(regime):
                continue
                
            # Strategy conditions
            price = current_bar["close"]
            price_below_vwap = price < current_bar["vwap"]
            obv_falling = current_bar["obv_diff"] < 0
            mfi_oversold = current_bar["mfi"] < strategy.mfi_entry
            
            if price_below_vwap and obv_falling and mfi_oversold:
                params = strategy.get_regime_parameters(regime)
                if not params:
                    continue
                    
                # Calculate position size
                size = self.calculate_position_size(
                    symbol, price, current_bar["atr"], params, vol_mult
                )
                
                if size > 0:
                    self._enter_position(
                        symbol, price, size, timestamp, regime, params,
                        atr=current_bar["atr"]
                    )
                    
    def _enter_position(self, symbol: str, price: float, size: int, 
                       timestamp: datetime, regime: str, params: Dict,
                       atr: float):
        """Enter new position"""
        position_cost = price * size
        if position_cost > self.capital:
            return
            
        # Calculate stops using provided ATR
        raw_stop = params["stop_mult"] * atr
        stop_dist = min(
            max(raw_stop, self.strategies[symbol].min_stop_dollars),
            self.strategies[symbol].max_stop_dollars
        )
        stop_loss = price - stop_dist
        take_profit = price + (stop_dist * params["reward_risk"])
        
        # Record position
        self.positions[symbol] = {
            "entry_time": timestamp,
            "entry_price": price,
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "regime": regime,
            "highest_price": price
        }
        
        # Update capital
        self.capital -= position_cost
        
        # Record trade
        self.record_trade(
            timestamp=timestamp,
            symbol=symbol,
            action="BUY",
            price=price,
            size=size,
            regime=regime
        )
        
    def _exit_position(self, symbol: str, price: float, timestamp: datetime, reason: str):
        """Exit existing position"""
        position = self.positions[symbol]
        
        # Calculate PnL
        pnl = (price - position["entry_price"]) * position["size"]
        
        # Update capital
        self.capital += (price * position["size"])
        
        # Record trade
        self.record_trade(
            timestamp=timestamp,
            symbol=symbol,
            action="SELL",
            price=price,
            size=position["size"],
            regime=position["regime"],
            reason=reason,
            pnl=pnl
        )
        
        # Remove position
        del self.positions[symbol]
        
    def run(self, data: Dict[str, pd.DataFrame], config: Dict) -> List[Dict]:
        """Run portfolio strategy on multiple symbols"""
        # Reset state
        self.capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.portfolio_history.clear()
        self.strategies.clear()
        self.symbol_cooldowns.clear()
        
        # Precompute indicators for each symbol
        processed_data = {}
        for symbol, df in data.items():
            # Get parameters
            regime_window = config.get("Regime Window", 20)
            volatility_percentile = config.get("Volatility Percentile", 67)
            
            # Precompute
            df_pre = precompute_all_indicators(
                df,
                regime_window=regime_window,
                volatility_percentile=volatility_percentile
            )
            
            # Fill regime
            df_reg = fill_regime_column(df_pre)
            
            processed_data[symbol] = df_reg
            
        # Get common index across all symbols
        common_index = None
        for df in processed_data.values():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
                
        # Process bars chronologically
        for timestamp in common_index:
            symbol_data = {
                symbol: df.loc[timestamp]
                for symbol, df in processed_data.items()
            }
            self.process_bar(timestamp, symbol_data)
            
        return self.trades

def run_portfolio_strategy(data: Dict[str, pd.DataFrame], config: Dict) -> List[Dict]:
    """Convenience function to run portfolio strategy"""
    portfolio = PortfolioManager(config)
    return portfolio.run(data, config)
