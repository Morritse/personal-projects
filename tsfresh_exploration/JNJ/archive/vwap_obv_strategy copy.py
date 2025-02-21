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
        self.morning_start = time(9, 30)
        self.afternoon_start = time(12, 0)
        self.market_close = time(16, 0)
        self.afterhours_end = time(20, 0)
        
        # Risk parameters
        self.stop_loss_atr = config.get('Stop Loss ATR', 1.2)
        self.min_profit_ratio = config.get('Min Profit Ratio', 1.5)
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
        vol_threshold = vol.median()
        
        if current_ret > 0:
            regime = 'bull_high_vol' if current_vol > vol_threshold else 'bull_low_vol'
        else:
            regime = 'bear_high_vol' if current_vol > vol_threshold else 'bear_low_vol'
        
        self.current_regime = regime
        return regime
    
    def get_regime_parameters(self, regime: str) -> Dict:
        """Get regime-specific parameters."""
        params = {
            'bull_high_vol': {
                'position_scale': 0.5,
                'stop_mult': 1.2
            },
            'bear_high_vol': {
                'position_scale': 1.0,
                'stop_mult': 1.2
            },
            'bull_low_vol': {
                'position_scale': 0.0,
                'stop_mult': 1.2
            },
            'bear_low_vol': {
                'position_scale': 0.5,
                'stop_mult': 1.2
            }
        }
        return params.get(regime, params['bear_high_vol'])
    
    def check_time_window(self, timestamp: datetime) -> bool:
        """Check if current time is in trading window."""
        current_time = timestamp.time()
        return (self.morning_start <= current_time < self.afterhours_end)
    
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
        if not self.check_time_window(timestamp):
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
        obv_falling = obv.diff().iloc[-1] < 0
        mfi_oversold = mfi.iloc[-1] < 30
        
        # All conditions must be true
        if price_below_vwap and obv_falling and mfi_oversold:
            atr = self.calculate_atr(jnj_data).iloc[-1]
            stop_loss = price - (params['stop_mult'] * atr)
            take_profit = price + (params['stop_mult'] * atr * 2)  # 2:1 reward/risk
            
            print(f"Entry signal generated:")
            print(f"- Price: ${price:.2f}")
            print(f"- Stop Loss: ${stop_loss:.2f}")
            print(f"- Take Profit: ${take_profit:.2f}")
            
            return True, {
                'action': 'BUY',
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'regime': regime,
                'entry_time': timestamp
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
                              regime: str) -> int:
        """Calculate position size based on risk and regime."""
        # Get regime parameters
        params = self.get_regime_parameters(regime)
        
        # Calculate base size from risk
        risk_amount = capital * self.risk_per_trade
        risk_per_share = atr * params['stop_mult']
        base_size = risk_amount / risk_per_share
        
        # Scale by regime
        position_size = round(base_size * params['position_scale'])
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
                        entry_data['regime']
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
    'Risk Per Trade': 0.01,
    'Max Hold Hours': 24
}
