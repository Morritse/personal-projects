import numpy as np
import talib
from dataclasses import dataclass
from typing import Dict

@dataclass
class MomentumSignal:
    normalized_signal: float
    confidence: float
    rsi_value: float
    stoch_rsi: float
    cci_value: float
    is_overbought: bool
    is_oversold: bool

class MomentumComponent:
    def __init__(
        self,
        weight: float = 0.3,
        rsi_period: int = 14,
        stoch_period: int = 14,
        cci_period: int = 20,
        min_periods: int = 30
    ):
        self.weight = weight
        self.rsi_period = rsi_period
        self.stoch_period = stoch_period
        self.cci_period = cci_period
        self.min_periods = min_periods
        
    def analyze(self, data: Dict[str, np.ndarray]) -> MomentumSignal:
        """Analyze momentum indicators and generate signal."""
        # Get data arrays
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Check data length
        if len(close) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} periods")
            
        # Calculate RSI
        rsi = talib.RSI(close, timeperiod=self.rsi_period)
        rsi_value = rsi[-1]
        
        # Calculate Stochastic RSI
        fastk, fastd = talib.STOCHRSI(close)
        stoch_rsi = fastk[-1]  # Use fast %K line
        
        # Calculate CCI
        cci = talib.CCI(high, low, close, timeperiod=self.cci_period)
        cci_value = cci[-1]
        
        # Determine overbought/oversold conditions
        is_overbought = (
            rsi_value > 70 and
            stoch_rsi > 0.8 and
            cci_value > 100
        )
        
        is_oversold = (
            rsi_value < 30 and
            stoch_rsi < 0.2 and
            cci_value < -100
        )
        
        # Calculate normalized signal (-1 to 1)
        rsi_signal = 2 * (rsi_value - 50) / 100  # Center around 0
        stoch_signal = 2 * (stoch_rsi - 0.5)     # Center around 0
        cci_signal = cci_value / 200             # Normalize to roughly -1 to 1
        
        # Combine signals with weights
        momentum_signal = np.clip(
            0.4 * rsi_signal +
            0.3 * stoch_signal +
            0.3 * cci_signal,
            -1, 1
        )
        
        # Calculate confidence based on agreement
        signals = np.array([rsi_signal, stoch_signal, cci_signal])
        signal_std = np.std(signals)
        agreement = 1 - min(signal_std, 0.5) / 0.5  # Higher agreement = lower std dev
        
        # Boost confidence in extreme conditions
        if is_overbought or is_oversold:
            confidence = min(1.0, agreement * 1.5)
        else:
            confidence = agreement
            
        return MomentumSignal(
            normalized_signal=momentum_signal,
            confidence=confidence,
            rsi_value=rsi_value,
            stoch_rsi=stoch_rsi,
            cci_value=cci_value,
            is_overbought=is_overbought,
            is_oversold=is_oversold
        )
