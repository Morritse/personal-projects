import pandas as pd
import numpy as np
import talib
from numba import njit
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Union
from config import config

class Indicators:
    """Module for indicator calculations"""
    
    @staticmethod
    def calculate_vwap(high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate VWAP using pandas rolling window"""
        typical_price = (high + low + close) / 3
        vwap = pd.Series(typical_price * volume).cumsum() / pd.Series(volume).cumsum()
        return vwap.values  # Convert back to numpy array
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate rolling metrics using pandas rolling window"""
        # Convert to pandas Series if numpy array
        if isinstance(returns, np.ndarray):
            returns = pd.Series(returns)
            
        # Calculate rolling metrics
        ret = returns.rolling(window=window, min_periods=1).mean() * 252
        vol = returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)
        
        return ret.values, vol.values

class RegimeClassifier:
    """Module for market regime classification"""
    
    def __init__(self, current_window: int = 16, historical_window: int = 400, volatility_multiplier: float = 1.0):
        """
        Initialize with windows and volatility threshold multiplier:
        - current_window: 16 bars (2 trading days) for current volatility
        - historical_window: 400 bars (50 trading days) for baseline volatility
        - volatility_multiplier: multiplier for std dev threshold (default 1.0)
        """
        self.current_window = current_window
        self.historical_window = historical_window
        self.volatility_multiplier = volatility_multiplier
        self.indicators = Indicators()
    
    def classify(self, data: pd.DataFrame) -> str:
        """
        Classify market regime using ATR volatility
        ATR as a percentage of price gives us a comparable volatility measure across symbols
        """
        if len(data) < max(self.current_window, self.historical_window):
            return None
            
        # Create DataFrame with OHLC data
        df = pd.DataFrame({
            'high': data['high'],
            'low': data['low'],
            'close': data['close']
        })
        
        # Calculate True Range components
        df['hl'] = df['high'] - df['low']
        df['hc'] = abs(df['high'] - df['close'].shift(1))
        df['lc'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
        df['tr_pct'] = (df['tr'] / df['close']) * 100
        
        # Calculate volatility metrics using rolling windows
        df['current_vol'] = df['tr_pct'].rolling(window=self.current_window, min_periods=1).mean()
        df['historical_vol'] = df['tr_pct'].rolling(window=self.historical_window, min_periods=1).mean()
        df['historical_std'] = df['tr_pct'].rolling(window=self.historical_window, min_periods=1).std()
        
        # Get latest values
        current_vol = df['current_vol'].iloc[-1]
        historical_mean = df['historical_vol'].iloc[-1]
        historical_std = df['historical_std'].iloc[-1]
        
        # Calculate trend direction
        current_ret = df['close'].pct_change().iloc[-1]
        
        # Consider volatile if current ATR% is above mean + (multiplier * std dev)
        vol_threshold = historical_mean + (historical_std * self.volatility_multiplier)
        if current_ret > 0 and current_vol > vol_threshold:
            return 'bull_high_vol'
        elif current_ret <= 0 and current_vol > vol_threshold:
            return 'bear_high_vol'
        return None

class PositionSizer:
    """Module for position sizing calculations"""
    
    def __init__(self, config: Dict):
        self.risk_per_trade = config.get('Risk Per Trade', 0.025)
        self.min_stop_dollars = config.get('Min Stop Dollars', 1.00)
        self.max_stop_dollars = config.get('Max Stop Dollars', 2.50)
    
    def calculate_size(self, price: float, atr: float, capital: float,
                     regime_params: Dict, vol_mult: float) -> int:
        """Calculate position size with risk management"""
        risk_amount = capital * self.risk_per_trade
        risk_per_share = atr * regime_params['stop_mult']
        base_size = risk_amount / risk_per_share
        position_size = round(base_size * regime_params['position_scale'] * vol_mult)
        return max(1, position_size)

class VAMEStrategy:  # Volatility Adaptive Momentum Engine
    def precompute_indicators(self, symbol_data: pd.DataFrame) -> pd.DataFrame:
        """Pre-compute all technical indicators to avoid recalculation in the loop"""
        df = symbol_data.copy()
        
        # VWAP
        df['vwap'] = self.indicators.calculate_vwap(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values
        )
        
        # OBV and its momentum
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['obv_diff'] = df['obv'].diff()
        
        # MFI
        df['mfi'] = talib.MFI(
            df['high'],
            df['low'],
            df['close'],
            df['volume'],
            timeperiod=self.mfi_period
        )
        
        # ATR
        df['atr'] = talib.ATR(
            df['high'],
            df['low'],
            df['close'],
            timeperiod=self.atr_period
        )
        
        return df
        
    def __init__(self, config: Dict):
        """Initialize strategy with configuration parameters."""
        # Initialize modules
        self.indicators = Indicators()
        self.regime_classifier = RegimeClassifier(
            current_window=config.get('Current Window', 16),
            historical_window=config.get('Historical Window', 400),
            volatility_multiplier=config.get('Volatility Multiplier', 1.0)
        )
        self.position_sizer = PositionSizer(config)
        
        # Core parameters
        self.mfi_period = config.get('MFI Period', 9)
        self.vwap_window = config.get('VWAP Window', 50)
        self.atr_period = config.get('ATR Period', 2)
        
        # MFI thresholds
        self.mfi_entry = config.get('mfi_entry', 30)
        self.bear_exit = config.get('bear_exit', 55)
        self.bull_exit = config.get('bull_exit', 75)
        
        # Time windows
        self.morning_start = time(9, 30)     # 9:30 AM
        self.midmorning = time(10, 30)       # 10:30 AM
        self.afternoon_start = time(12, 0)   # 12:00 PM
        self.market_close = time(16, 0)      # 4:00 PM
        
        # Risk parameters
        self.min_stop_dollars = config.get('Min Stop Dollars', 1.00)
        self.max_stop_dollars = config.get('Max Stop Dollars', 2.50)
        self.risk_per_trade = config.get('Risk Per Trade', 0.025)
        self.max_hold_hours = config.get('Max Hold Hours', 36)
        
        # Regime parameters (only high vol)
        self.regime_params = config.get('regime_params', {
            'bear_high_vol': {
                'position_scale': 2.25,
                'reward_risk': 2.0,
                'stop_mult': 1.6,
                'mfi_overbought': self.bear_exit,
                'trailing_stop': True
            },
            'bull_high_vol': {
                'position_scale': 1.50,
                'reward_risk': 2.0,
                'stop_mult': 1.6,
                'mfi_overbought': self.bull_exit,
                'trailing_stop': True
            }
        })
        
        # State
        self.position = None
        self.trades = []
        self.current_regime = None
        self.highest_price = None
    
    def get_regime_parameters(self, regime: str) -> Dict:
        """Get regime-specific parameters."""
        return self.regime_params.get(regime, {})
    
    def check_time_window(self, timestamp: datetime) -> Tuple[bool, float]:
        """Check if current time is in trading window and return volatility multiplier."""
        current_time = timestamp.time()
        
        if current_time < self.morning_start:
            return False, 0.0  # Skip pre-market
        elif self.morning_start <= current_time < self.midmorning:
            return True, 0.8   # Reduced size during volatile open
        elif self.midmorning <= current_time < self.afternoon_start:
            return True, 1.2   # Best trading window
        elif self.afternoon_start <= current_time < self.market_close:
            return True, 1.0   # Normal trading
        else:
            return False, 0.0  # Skip after-hours
    
    def check_entry(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """Check entry conditions using precomputed indicators."""
        if len(data) < 20:
            return False, None
        
        # Get current timestamp and check time window
        timestamp = data.index[-1]
        tradeable, vol_mult = self.check_time_window(timestamp)
        if not tradeable:
            return False, None
        
        # Classify regime using symbol's own volatility
        regime = self.regime_classifier.classify(data)
        if not regime:  # Skip if not high volatility
            return False, None
            
        params = self.get_regime_parameters(regime)
        if not params:  # Skip if regime not configured
            return False, None
        
        # Use precomputed indicators
        price = data['close'].iloc[-1]
        price_below_vwap = price < data['vwap'].iloc[-1]
        obv_falling = data['obv_diff'].iloc[-1] < 0
        mfi_oversold = data['mfi'].iloc[-1] < self.mfi_entry
        
        # All conditions must be true
        if price_below_vwap and obv_falling and mfi_oversold:
            atr = talib.ATR(
                symbol_data['high'],
                symbol_data['low'],
                symbol_data['close'],
                timeperiod=self.atr_period
            ).iloc[-1]
            
            # Calculate position size
            size = self.position_sizer.calculate_size(
                price, atr, 100000, params, vol_mult
            )
            
            # Calculate stop loss with floor/ceiling
            raw_stop = params['stop_mult'] * atr
            stop_distance = min(max(raw_stop, self.min_stop_dollars), self.max_stop_dollars)
            stop_loss = price - stop_distance
            
            # Dynamic reward/risk based on regime
            take_profit = price + (stop_distance * params['reward_risk'])
            
            # Initialize highest price for trailing stop
            self.highest_price = price if params.get('trailing_stop', False) else None
            
            return True, {
                'action': 'BUY',
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'regime': regime,
                'entry_time': timestamp,
                'vol_mult': vol_mult,
                'size': size
            }
        
        return False, None
    
    def check_exit(self, data: pd.DataFrame, position: Dict) -> Tuple[bool, Dict]:
        """Check exit conditions using precomputed indicators."""
        if not position:
            return False, None
        
        current_time = data.index[-1]
        price = data['close'].iloc[-1]
        
        # Get regime parameters
        regime = position['regime']
        params = self.get_regime_parameters(regime)
        
        # Update trailing stop if enabled
        if params.get('trailing_stop', False) and self.highest_price is not None:
            self.highest_price = max(self.highest_price, price)
            trailing_stop = self.highest_price - (position['take_profit'] - position['entry_price']) * 0.5
            position['stop_loss'] = max(position['stop_loss'], trailing_stop)
        
        # Check holding period
        hours_held = (current_time - position['entry_time']).total_seconds() / 3600
        if hours_held > self.max_hold_hours:
            return True, {
                'action': 'SELL',
                'price': price,
                'reason': 'max_hold_time'
            }
        
        # Stop loss or take profit hit
        if price <= position['stop_loss'] or price >= position['take_profit']:
            return True, {
                'action': 'SELL',
                'price': price,
                'reason': 'stop_or_target'
            }
        
        # Use precomputed indicators
        price_above_vwap = price > data['vwap'].iloc[-1]
        mfi_overbought = data['mfi'].iloc[-1] > params.get('mfi_overbought', 70)
        
        # Exit if price above VWAP or MFI overbought
        if price_above_vwap or mfi_overbought:
            return True, {
                'action': 'SELL',
                'price': price,
                'reason': 'technical'
            }
        
        return False, None
    
    def run(self, symbol_data: pd.DataFrame) -> List[Dict]:
        """Run strategy."""
        self.trades = []
        self.position = None
        self.highest_price = None
        
        try:
            # Pre-compute all indicators
            df = self.precompute_indicators(symbol_data)
            
            # Need enough data for volatility calculation (400 bars) plus VWAP window
            min_window = max(400, self.vwap_window)
            
            # Skip first 400 bars to establish baseline volatility
            for i in range(min_window, len(df)):
                # Get current bar data
                current_bar = df.iloc[i]
                
                # Create view for regime classification
                window_data = df.iloc[max(0, i-400):i+1]
                
                if not self.position:
                    # Entry conditions
                    price = current_bar['close']
                    price_below_vwap = price < current_bar['vwap']
                    obv_falling = current_bar['obv_diff'] < 0
                    mfi_oversold = current_bar['mfi'] < self.mfi_entry
                    
                    # Check regime
                    regime = self.regime_classifier.classify(window_data)
                    if not regime:
                        continue
                        
                    # Check time window
                    tradeable, vol_mult = self.check_time_window(current_bar.name)
                    if not tradeable:
                        continue
                    
                    # All conditions must be true
                    if price_below_vwap and obv_falling and mfi_oversold:
                        params = self.get_regime_parameters(regime)
                        if not params:
                            continue
                            
                        # Calculate position size
                        size = self.position_sizer.calculate_size(
                            price, current_bar['atr'], 100000, params, vol_mult
                        )
                        
                        # Calculate stop loss with floor/ceiling
                        raw_stop = params['stop_mult'] * current_bar['atr']
                        stop_distance = min(max(raw_stop, self.min_stop_dollars), self.max_stop_dollars)
                        stop_loss = price - stop_distance
                        
                        # Dynamic reward/risk based on regime
                        take_profit = price + (stop_distance * params['reward_risk'])
                        
                        # Initialize highest price for trailing stop
                        self.highest_price = price if params.get('trailing_stop', False) else None
                        
                        # Record entry
                        self.position = {
                            'entry_time': current_bar.name,
                            'entry_price': price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'size': size,
                            'regime': regime
                        }
                        
                        self.trades.append({
                            'timestamp': current_bar.name,
                            'action': 'BUY',
                            'price': price,
                            'size': size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'regime': regime
                        })
                
                else:
                    # Exit conditions
                    price = current_bar['close']
                    
                    # Update trailing stop if enabled
                    if self.position['regime'] in self.regime_params and self.highest_price is not None:
                        self.highest_price = max(self.highest_price, price)
                        trailing_stop = self.highest_price - (self.position['take_profit'] - self.position['entry_price']) * 0.5
                        self.position['stop_loss'] = max(self.position['stop_loss'], trailing_stop)
                    
                    # Check stop loss/take profit
                    if price <= self.position['stop_loss'] or price >= self.position['take_profit']:
                        pnl = (price - self.position['entry_price']) * self.position['size']
                        
                        self.trades.append({
                            'timestamp': current_bar.name,
                            'action': 'SELL',
                            'price': price,
                            'size': self.position['size'],
                            'pnl': pnl,
                            'reason': 'stop_or_target',
                            'regime': self.position['regime']
                        })
                        
                        self.position = None
                        self.highest_price = None
                        continue
                    
                    # Check technical exit conditions
                    price_above_vwap = price > current_bar['vwap']
                    mfi_overbought = current_bar['mfi'] > self.bull_exit
                    
                    if price_above_vwap or mfi_overbought:
                        pnl = (price - self.position['entry_price']) * self.position['size']
                        
                        self.trades.append({
                            'timestamp': current_bar.name,
                            'action': 'SELL',
                            'price': price,
                            'size': self.position['size'],
                            'pnl': pnl,
                            'reason': 'technical',
                            'regime': self.position['regime']
                        })
                        
                        self.position = None
                        self.highest_price = None
                        
        except Exception as e:
            print(f"Error in strategy execution: {e}")
        
        return self.trades
