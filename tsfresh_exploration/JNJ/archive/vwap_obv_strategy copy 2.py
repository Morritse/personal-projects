import pandas as pd
import numpy as np
import talib
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Union

class VWAPOBVCrossover:
    def __init__(self, config: Dict):
        """Initialize strategy with configuration parameters."""
        # Core parameters
        self.mfi_period = config.get('MFI Period', 9)
        self.vwap_window = config.get('VWAP Window', 120)  # 2 hours
        self.atr_period = config.get('ATR Period', 3)
        self.regime_window = config.get('Regime Window', 20)
        
        # Time windows
        self.premarket_start = time(8, 0)    # 8:00 AM
        self.morning_start = time(9, 30)     # 9:30 AM
        self.midmorning = time(10, 30)       # 10:30 AM
        self.afternoon_start = time(12, 0)   # 12:00 PM
        self.market_close = time(16, 0)      # 4:00 PM
        
        # Risk parameters
        self.stop_loss_atr = config.get('Stop Loss ATR', 1.2)
        self.min_stop_dollars = config.get('Min Stop Dollars', 0.50)  # Minimum stop distance
        self.max_stop_dollars = config.get('Max Stop Dollars', 2.00)  # Maximum stop distance
        self.risk_per_trade = config.get('Risk Per Trade', 0.01)
        self.max_hold_hours = config.get('Max Hold Hours', 24)
        
        # State
        self.position = None
        self.trades = []
        self.current_regime = None
    
    def classify_regime(self, data: pd.DataFrame) -> str:
        """Classify market regime based on returns and volatility."""
        returns = data['close'].pct_change()
        
        # Calculate rolling metrics
        ret = returns.rolling(window=self.regime_window).mean() * 252
        vol = returns.rolling(window=self.regime_window).std() * np.sqrt(252)
        
        # Get current values
        current_ret = ret.iloc[-1]
        current_vol = vol.iloc[-1]
        
        # Use quantile-based thresholds for volatility
        vol_33pct = vol.quantile(0.33)
        vol_67pct = vol.quantile(0.67)
        
        # Classify regime
        if current_ret > 0:
            if current_vol > vol_67pct:
                regime = 'bull_high_vol'
            elif current_vol < vol_33pct:
                regime = 'bull_low_vol'
            else:
                regime = 'bull_med_vol'
        else:
            if current_vol > vol_67pct:
                regime = 'bear_high_vol'
            elif current_vol < vol_33pct:
                regime = 'bear_low_vol'
            else:
                regime = 'bear_med_vol'
        
        self.current_regime = regime
        return regime
    
    def get_regime_parameters(self, regime: str) -> Dict:
        """Get regime-specific parameters."""
        params = {
            'bull_high_vol': {
                'position_scale': 0.5,    # Reduced size
                'stop_mult': 1.2,
                'reward_risk': 2.0        # Standard R:R
            },
            'bull_med_vol': {
                'position_scale': 0.3,    # Small size
                'stop_mult': 1.2,
                'reward_risk': 1.5        # Reduced R:R
            },
            'bull_low_vol': {
                'position_scale': 0.0,    # Skip
                'stop_mult': 1.2,
                'reward_risk': 1.5
            },
            'bear_high_vol': {
                'position_scale': 1.5,    # Increased size (best performer)
                'stop_mult': 1.2,
                'reward_risk': 2.5        # Higher R:R
            },
            'bear_med_vol': {
                'position_scale': 0.5,    # Reduced size
                'stop_mult': 1.2,
                'reward_risk': 2.0
            },
            'bear_low_vol': {
                'position_scale': 0.0,    # Skip (losing regime)
                'stop_mult': 1.2,
                'reward_risk': 1.5
            }
        }
        return params.get(regime, params['bear_high_vol'])
    
    def check_time_window(self, timestamp: datetime) -> Tuple[bool, float]:
        """Check if current time is in trading window and return volatility multiplier."""
        current_time = timestamp.time()
        
        if self.premarket_start <= current_time < self.morning_start:
            return False, 0.0  # Skip pre-market
        elif self.morning_start <= current_time < self.midmorning:
            return True, 0.8   # Reduced size during volatile open
        elif self.midmorning <= current_time < self.afternoon_start:
            return True, 1.2   # Best trading window
        elif self.afternoon_start <= current_time < self.market_close:
            return True, 1.0   # Normal trading
        else:
            return False, 0.0  # Skip after-hours
    
    def calculate_vwap(self, data: pd.DataFrame, window: Optional[int] = None) -> pd.Series:
        """Calculate VWAP for specified window."""
        if window:
            data = data.iloc[-window:]
        
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        return vwap
    
    def calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        return talib.OBV(data['close'], data['volume'])
    
    def calculate_mfi(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Money Flow Index."""
        return talib.MFI(data['high'], data['low'], data['close'], 
                        data['volume'], timeperiod=self.mfi_period)
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        return talib.ATR(data['high'], data['low'], data['close'], 
                        timeperiod=self.atr_period)
    
    def check_entry(self, jnj_data: pd.DataFrame, xlv_data: pd.DataFrame) -> Tuple[bool, Dict]:
        """Check entry conditions."""
        if len(jnj_data) < 20:
            return False, None
        
        # Get current timestamp and check time window
        timestamp = jnj_data.index[-1]
        tradeable, vol_mult = self.check_time_window(timestamp)
        if not tradeable:
            print(f"Time window not tradeable: {timestamp.time()}")
            return False, None
        
        # Classify regime and get parameters
        regime = self.classify_regime(xlv_data)
        params = self.get_regime_parameters(regime)
        
        # Skip if regime not favorable
        if params['position_scale'] == 0.0:
            print(f"Skipping unfavorable regime: {regime}")
            return False, None
        
        # Calculate indicators
        vwap = self.calculate_vwap(jnj_data, self.vwap_window)
        obv = self.calculate_obv(jnj_data)
        mfi = self.calculate_mfi(jnj_data)
        
        # Entry conditions
        price = jnj_data['close'].iloc[-1]
        price_below_vwap = price < vwap.iloc[-1]
        price_to_vwap = (price / vwap.iloc[-1] - 1) * 100
        obv_falling = obv.diff().iloc[-1] < 0
        mfi_oversold = mfi.iloc[-1] < 30
        
        # All conditions must be true
        if price_below_vwap and obv_falling and mfi_oversold:
            atr = self.calculate_atr(jnj_data).iloc[-1]
            
            # Calculate stop loss with floor/ceiling
            raw_stop = params['stop_mult'] * atr
            stop_distance = min(max(raw_stop, self.min_stop_dollars), self.max_stop_dollars)
            stop_loss = price - stop_distance
            
            # Dynamic reward/risk based on regime
            take_profit = price + (stop_distance * params['reward_risk'])
            
            print(f"\nEntry signal generated:")
            print(f"Regime: {regime}")
            print(f"Time: {timestamp.time()}")
            print(f"Price: ${price:.2f}")
            print(f"VWAP Distance: {price_to_vwap:.2f}%")
            print(f"MFI: {mfi.iloc[-1]:.1f}")
            print(f"Stop Loss: ${stop_loss:.2f} (${stop_distance:.2f} distance)")
            print(f"Take Profit: ${take_profit:.2f} ({params['reward_risk']}:1 R:R)")
            
            return True, {
                'action': 'BUY',
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'regime': regime,
                'entry_time': timestamp,
                'vol_mult': vol_mult
            }
        
        return False, None
    
    def check_exit(self, jnj_data: pd.DataFrame, xlv_data: pd.DataFrame, 
                   position: Dict) -> Tuple[bool, Dict]:
        """Check exit conditions."""
        if not position:
            return False, None
        
        current_time = jnj_data.index[-1]
        price = jnj_data['close'].iloc[-1]
        
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
        
        # Check technical conditions
        vwap = self.calculate_vwap(jnj_data, self.vwap_window)
        mfi = self.calculate_mfi(jnj_data)
        
        # Exit if price above VWAP or MFI overbought
        if price > vwap.iloc[-1] or mfi.iloc[-1] > 70:
            return True, {
                'action': 'SELL',
                'price': price,
                'reason': 'technical'
            }
        
        return False, None
    
    def calculate_position_size(self, price: float, atr: float, capital: float, 
                              regime: str, vol_mult: float) -> int:
        """Calculate position size based on risk, regime, and time of day."""
        # Get regime parameters
        params = self.get_regime_parameters(regime)
        
        # Calculate base size from risk
        risk_amount = capital * self.risk_per_trade
        risk_per_share = atr * params['stop_mult']
        base_size = risk_amount / risk_per_share
        
        # Scale by regime and time of day
        position_size = round(base_size * params['position_scale'] * vol_mult)
        return max(1, position_size)  # Minimum 1 share
    
    def run(self, jnj_data: pd.DataFrame, xlv_data: pd.DataFrame, 
            capital: float = 100000) -> List[Dict]:
        """Run strategy."""
        self.trades = []
        self.position = None
        
        for i in range(20, len(jnj_data)):
            window_jnj = jnj_data.iloc[max(0, i-50):i+1].copy()
            window_xlv = xlv_data.iloc[max(0, i-50):i+1].copy()
            current_bar = window_jnj.iloc[-1]
            
            if not self.position:
                # Check for entry
                entry_signal, entry_data = self.check_entry(window_jnj, window_xlv)
                if entry_signal:
                    atr = self.calculate_atr(window_jnj).iloc[-1]
                    size = self.calculate_position_size(
                        entry_data['price'], atr, capital,
                        entry_data['regime'], entry_data['vol_mult']
                    )
                    
                    self.position = {
                        'entry_time': entry_data['entry_time'],
                        'entry_price': entry_data['price'],
                        'stop_loss': entry_data['stop_loss'],
                        'take_profit': entry_data['take_profit'],
                        'size': size,
                        'regime': entry_data['regime']
                    }
                    
                    self.trades.append({
                        'timestamp': current_bar.name,
                        'action': 'BUY',
                        'price': entry_data['price'],
                        'size': size,
                        'stop_loss': entry_data['stop_loss'],
                        'take_profit': entry_data['take_profit'],
                        'regime': entry_data['regime']
                    })
            
            else:
                # Check for exit
                exit_signal, exit_data = self.check_exit(window_jnj, window_xlv, self.position)
                if exit_signal:
                    pnl = (exit_data['price'] - self.position['entry_price']) * self.position['size']
                    
                    self.trades.append({
                        'timestamp': current_bar.name,
                        'action': 'SELL',
                        'price': exit_data['price'],
                        'size': self.position['size'],
                        'pnl': pnl,
                        'reason': exit_data['reason'],
                        'regime': self.position['regime']
                    })
                    
                    self.position = None
        
        return self.trades

# Strategy Configuration
strategy_configuration = {
    'MFI Period': 9,
    'VWAP Window': 120,
    'ATR Period': 3,
    'Regime Window': 20,
    'Min Stop Dollars': 0.50,
    'Max Stop Dollars': 2.00,
    'Risk Per Trade': 0.01,
    'Max Hold Hours': 24
}
