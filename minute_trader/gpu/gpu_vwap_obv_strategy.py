import pandas as pd
import numpy as np
import cupy as cp
import talib
from numba import cuda
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Union
from config import config

@cuda.jit
def calculate_vwap_kernel(high, low, close, volume, result):
    """CUDA kernel for VWAP calculation"""
    idx = cuda.grid(1)
    if idx < high.shape[0]:
        # Calculate typical price
        typical_price = (high[idx] + low[idx] + close[idx]) / 3.0
        # Multiply by volume
        result[idx] = typical_price * volume[idx]

class GPUIndicators:
    """GPU-accelerated indicator calculations"""
    
    @staticmethod
    def calculate_vwap(high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Calculate VWAP using CUDA acceleration"""
        # Transfer data to GPU
        high_gpu = cuda.to_device(high)
        low_gpu = cuda.to_device(low)
        close_gpu = cuda.to_device(close)
        volume_gpu = cuda.to_device(volume)
        result_gpu = cuda.device_array_like(high)
        
        # Configure CUDA grid
        threadsperblock = 256
        blockspergrid = (high.shape[0] + (threadsperblock - 1)) // threadsperblock
        
        # Launch kernel
        calculate_vwap_kernel[blockspergrid, threadsperblock](
            high_gpu, low_gpu, close_gpu, volume_gpu, result_gpu
        )
        
        # Get result back to CPU
        result = result_gpu.copy_to_host()
        
        # Calculate cumulative sums on CPU (faster for small arrays)
        vwap = pd.Series(result).cumsum() / pd.Series(volume).cumsum()
        return vwap.values
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate rolling metrics using CuPy"""
        # Convert to CuPy array
        returns_gpu = cp.asarray(returns)
        
        # Calculate rolling metrics on GPU
        ret = cp.zeros_like(returns_gpu)
        vol = cp.zeros_like(returns_gpu)
        
        for i in range(window - 1, len(returns_gpu)):
            window_data = returns_gpu[i-window+1:i+1]
            ret[i] = cp.mean(window_data) * 252
            vol[i] = cp.std(window_data) * cp.sqrt(252)
        
        # Transfer back to CPU
        return cp.asnumpy(ret), cp.asnumpy(vol)

class GPURegimeClassifier:
    """GPU-accelerated market regime classification"""
    
    def __init__(self, current_window: int = 16, historical_window: int = 400, volatility_multiplier: float = 1.0):
        self.current_window = current_window
        self.historical_window = historical_window
        self.volatility_multiplier = volatility_multiplier
        self.indicators = GPUIndicators()
    
    def classify(self, data: pd.DataFrame) -> str:
        """Classify market regime using GPU-accelerated calculations"""
        if len(data) < max(self.current_window, self.historical_window):
            return None
            
        # Transfer data to GPU
        high_gpu = cp.asarray(data['high'].values)
        low_gpu = cp.asarray(data['low'].values)
        close_gpu = cp.asarray(data['close'].values)
        
        # Calculate True Range components on GPU
        hl = high_gpu - low_gpu
        hc = cp.abs(high_gpu - cp.roll(close_gpu, 1))
        lc = cp.abs(low_gpu - cp.roll(close_gpu, 1))
        tr = cp.maximum(cp.maximum(hl, hc), lc)
        tr_pct = (tr / close_gpu) * 100
        
        # Calculate volatility metrics
        current_vol = cp.mean(tr_pct[-self.current_window:])
        historical_vol = cp.mean(tr_pct[-self.historical_window:])
        historical_std = cp.std(tr_pct[-self.historical_window:])
        
        # Calculate trend direction
        current_ret = (close_gpu[-1] - close_gpu[-2]) / close_gpu[-2]
        
        # Transfer results back to CPU
        current_vol = float(current_vol)
        historical_mean = float(historical_vol)
        historical_std = float(historical_std)
        current_ret = float(current_ret)
        
        # Classify regime
        vol_threshold = historical_mean + (historical_std * self.volatility_multiplier)
        if current_ret > 0 and current_vol > vol_threshold:
            return 'bull_high_vol'
        elif current_ret <= 0 and current_vol > vol_threshold:
            return 'bear_high_vol'
        return None

class GPUVAMEStrategy:
    """GPU-accelerated Volatility Adaptive Momentum Engine"""
    
    def __init__(self, config: Dict):
        # Initialize modules with GPU support
        self.indicators = GPUIndicators()
        self.regime_classifier = GPURegimeClassifier(
            current_window=config.get('Current Window', 16),
            historical_window=config.get('Historical Window', 400),
            volatility_multiplier=config.get('Volatility Multiplier', 1.0)
        )
        
        # Keep other parameters the same
        self.mfi_period = config.get('MFI Period', 9)
        self.vwap_window = config.get('VWAP Window', 50)
        self.atr_period = config.get('ATR Period', 2)
        self.mfi_entry = config.get('mfi_entry', 30)
        self.bear_exit = config.get('bear_exit', 55)
        self.bull_exit = config.get('bull_exit', 75)
        self.morning_start = time(9, 30)
        self.midmorning = time(10, 30)
        self.afternoon_start = time(12, 0)
        self.market_close = time(16, 0)
        self.min_stop_dollars = config.get('Min Stop Dollars', 1.00)
        self.max_stop_dollars = config.get('Max Stop Dollars', 2.50)
        self.risk_per_trade = config.get('Risk Per Trade', 0.025)
        self.max_hold_hours = config.get('Max Hold Hours', 36)
        
        # Regime parameters
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
        return self.regime_params.get(regime, {})
    
    def check_time_window(self, timestamp: datetime) -> Tuple[bool, float]:
        current_time = timestamp.time()
        
        if current_time < self.morning_start:
            return False, 0.0
        elif self.morning_start <= current_time < self.midmorning:
            return True, 0.8
        elif self.midmorning <= current_time < self.afternoon_start:
            return True, 1.2
        elif self.afternoon_start <= current_time < self.market_close:
            return True, 1.0
        else:
            return False, 0.0
    
    def check_entry(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """Check entry conditions using GPU-accelerated calculations"""
        if len(data) < 20:
            return False, None
        
        timestamp = data.index[-1]
        tradeable, vol_mult = self.check_time_window(timestamp)
        if not tradeable:
            return False, None
        
        regime = self.regime_classifier.classify(data)
        if not regime:
            return False, None
            
        params = self.get_regime_parameters(regime)
        if not params:
            return False, None
        
        # Calculate indicators using GPU
        vwap = self.indicators.calculate_vwap(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values
        )
        
        # Some indicators still use TA-Lib (CPU) as they're already optimized
        obv = talib.OBV(data['close'], data['volume'])
        mfi = talib.MFI(
            data['high'],
            data['low'],
            data['close'],
            data['volume'],
            timeperiod=self.mfi_period
        )
        atr = talib.ATR(
            data['high'],
            data['low'],
            data['close'],
            timeperiod=self.atr_period
        )
        
        price = data['close'].iloc[-1]
        price_below_vwap = price < vwap[-1]
        obv_falling = obv.diff().iloc[-1] < 0
        mfi_oversold = mfi.iloc[-1] < self.mfi_entry
        
        if price_below_vwap and obv_falling and mfi_oversold:
            # Calculate position size and stops
            risk_amount = 100000 * self.risk_per_trade
            risk_per_share = atr.iloc[-1] * params['stop_mult']
            base_size = risk_amount / risk_per_share
            size = round(base_size * params['position_scale'] * vol_mult)
            size = max(1, size)
            
            raw_stop = params['stop_mult'] * atr.iloc[-1]
            stop_distance = min(max(raw_stop, self.min_stop_dollars), self.max_stop_dollars)
            stop_loss = price - stop_distance
            take_profit = price + (stop_distance * params['reward_risk'])
            
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
        """Check exit conditions using GPU-accelerated calculations"""
        if not position:
            return False, None
        
        current_time = data.index[-1]
        price = data['close'].iloc[-1]
        regime = position['regime']
        params = self.get_regime_parameters(regime)
        
        if params.get('trailing_stop', False) and self.highest_price is not None:
            self.highest_price = max(self.highest_price, price)
            trailing_stop = self.highest_price - (position['take_profit'] - position['entry_price']) * 0.5
            position['stop_loss'] = max(position['stop_loss'], trailing_stop)
        
        hours_held = (current_time - position['entry_time']).total_seconds() / 3600
        if hours_held > self.max_hold_hours:
            return True, {
                'action': 'SELL',
                'price': price,
                'reason': 'max_hold_time'
            }
        
        if price <= position['stop_loss'] or price >= position['take_profit']:
            return True, {
                'action': 'SELL',
                'price': price,
                'reason': 'stop_or_target'
            }
        
        # Calculate indicators using GPU
        vwap = self.indicators.calculate_vwap(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values
        )
        
        mfi = talib.MFI(
            data['high'],
            data['low'],
            data['close'],
            data['volume'],
            timeperiod=self.mfi_period
        )
        
        price_above_vwap = price > vwap[-1]
        mfi_overbought = mfi.iloc[-1] > params.get('mfi_overbought', 70)
        
        if price_above_vwap or mfi_overbought:
            return True, {
                'action': 'SELL',
                'price': price,
                'reason': 'technical'
            }
        
        return False, None
    
    def run(self, symbol_data: pd.DataFrame) -> List[Dict]:
        """Run strategy with GPU acceleration"""
        self.trades = []
        self.position = None
        self.highest_price = None
        
        try:
            min_window = max(400, self.vwap_window)
            
            for i in range(min_window, len(symbol_data)):
                window_data = symbol_data.iloc[max(0, i-400):i+1]
                
                if not self.position:
                    entry_signal, entry_data = self.check_entry(window_data)
                    if entry_signal:
                        self.position = {
                            'entry_time': entry_data['entry_time'],
                            'entry_price': entry_data['price'],
                            'stop_loss': entry_data['stop_loss'],
                            'take_profit': entry_data['take_profit'],
                            'size': entry_data['size'],
                            'regime': entry_data['regime']
                        }
                        
                        self.trades.append({
                            'timestamp': window_data.index[-1],
                            'action': 'BUY',
                            'price': entry_data['price'],
                            'size': entry_data['size'],
                            'stop_loss': entry_data['stop_loss'],
                            'take_profit': entry_data['take_profit'],
                            'regime': entry_data['regime']
                        })
                else:
                    exit_signal, exit_data = self.check_exit(window_data, self.position)
                    if exit_signal:
                        pnl = (exit_data['price'] - self.position['entry_price']) * self.position['size']
                        
                        self.trades.append({
                            'timestamp': window_data.index[-1],
                            'action': 'SELL',
                            'price': exit_data['price'],
                            'size': self.position['size'],
                            'pnl': pnl,
                            'reason': exit_data['reason'],
                            'regime': self.position['regime']
                        })
                        
                        self.position = None
                        self.highest_price = None
                        
        except Exception as e:
            print(f"Error in strategy execution: {e}")
            # Clean up GPU memory
            cp.get_default_memory_pool().free_all_blocks()
        
        return self.trades
