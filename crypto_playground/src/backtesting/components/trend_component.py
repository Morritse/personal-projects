import numpy as np
import talib
from dataclasses import dataclass
from typing import Dict

@dataclass
class TrendSignal:
    normalized_signal: float
    confidence: float
    adx_strength: float
    dmi_signal: float
    macd_signal: float
    is_strong_trend: bool

class TrendComponent:
    def __init__(
        self,
        weight: float = 0.4,
        adx_period: int = 14,
        dmi_period: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        min_periods: int = 30
    ):
        self.weight = weight
        self.adx_period = adx_period
        self.dmi_period = dmi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.min_periods = min_periods
        
    def analyze(self, data: Dict[str, np.ndarray]) -> TrendSignal:
        """Analyze trend indicators and generate signal."""
        # Get data arrays
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Check data length
        if len(close) < self.min_periods:
            raise ValueError(f"Need at least {self.min_periods} periods, got {len(close)}")
            
        # Calculate ADX
        adx = talib.ADX(high, low, close, timeperiod=self.adx_period)
        adx_strength = adx[-1]
        is_strong_trend = adx_strength > 25
        
        # Calculate DMI
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.dmi_period)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.dmi_period)
        dmi_signal = (plus_di[-1] - minus_di[-1]) / (plus_di[-1] + minus_di[-1])
        
        # Calculate MACD
        macd, signal, hist = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        macd_signal = hist[-1] / close[-1]  # Normalize by price
        
        # Calculate normalized signal
        trend_signal = np.clip(
            0.4 * np.sign(dmi_signal) * (abs(dmi_signal) ** 0.5) +
            0.4 * np.sign(macd_signal) * (abs(macd_signal) ** 0.5) +
            0.2 * (adx_strength / 100.0),
            -1, 1
        )
        
        # Calculate confidence
        confidence = min(1.0, (
            0.5 * (adx_strength / 50.0) +  # ADX confidence
            0.3 * abs(dmi_signal) +        # DMI confidence
            0.2 * abs(macd_signal)         # MACD confidence
        ))
        
        return TrendSignal(
            normalized_signal=trend_signal,
            confidence=confidence,
            adx_strength=adx_strength,
            dmi_signal=dmi_signal,
            macd_signal=macd_signal,
            is_strong_trend=is_strong_trend
        )
