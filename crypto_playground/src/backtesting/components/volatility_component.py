import numpy as np
import talib
from dataclasses import dataclass
from typing import Dict

@dataclass
class VolatilitySignal:
    normalized_signal: float
    confidence: float
    bb_width: float
    atr_value: float
    squeeze_strength: float
    is_squeeze: bool
    breakout_direction: int  # 1 for up, -1 for down, 0 for none

class VolatilityComponent:
    def __init__(
        self,
        weight: float = 0.3,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5,
        atr_period: int = 14,
        min_periods: int = 30
    ):
        self.weight = weight
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_mult = kc_mult
        self.atr_period = atr_period
        self.min_periods = min_periods
        
    def analyze(self, data: Dict[str, np.ndarray]) -> VolatilitySignal:
        """Analyze volatility indicators and generate signal."""
        # Get data arrays
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Check data length
        if len(close) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} periods")
            
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )
        
        # Calculate BB width
        bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
        
        # Calculate ATR
        atr = talib.ATR(high, low, close, timeperiod=self.atr_period)
        atr_value = atr[-1]
        
        # Calculate Keltner Channels
        typical_price = (high + low + close) / 3
        kc_middle = talib.SMA(typical_price, timeperiod=self.kc_period)
        range_ema = talib.EMA(high - low, timeperiod=self.kc_period)
        kc_upper = kc_middle + (range_ema * self.kc_mult)
        kc_lower = kc_middle - (range_ema * self.kc_mult)
        
        # Check for squeeze (BB inside KC)
        is_squeeze = (
            bb_upper[-1] <= kc_upper[-1] and
            bb_lower[-1] >= kc_lower[-1]
        )
        
        # Calculate squeeze strength
        if is_squeeze:
            squeeze_strength = 1.0 - (bb_width / ((kc_upper[-1] - kc_lower[-1]) / kc_middle[-1]))
        else:
            squeeze_strength = 0.0
            
        # Determine breakout direction
        if close[-1] > bb_upper[-1]:
            breakout_direction = 1
        elif close[-1] < bb_lower[-1]:
            breakout_direction = -1
        else:
            breakout_direction = 0
            
        # Calculate normalized signal
        volatility_signal = np.clip(
            breakout_direction * (1 - bb_width),  # Stronger signal in tight ranges
            -1, 1
        )
        
        # Calculate confidence
        base_confidence = 0.5 + (0.5 * squeeze_strength)  # Higher in squeeze
        
        # Boost confidence on breakouts
        if breakout_direction != 0:
            confidence = min(1.0, base_confidence * 1.5)
        else:
            confidence = base_confidence
            
        return VolatilitySignal(
            normalized_signal=volatility_signal,
            confidence=confidence,
            bb_width=bb_width,
            atr_value=atr_value,
            squeeze_strength=squeeze_strength,
            is_squeeze=is_squeeze,
            breakout_direction=breakout_direction
        )
