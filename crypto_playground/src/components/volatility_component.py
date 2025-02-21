import numpy as np
from typing import Dict
from dataclasses import dataclass
from .base_component import BaseComponent, SignalResult

@dataclass
class VolatilitySignalResult(SignalResult):
    bb_width: float = 0.0
    atr_value: float = 0.0
    squeeze_strength: float = 0.0

class VolatilityComponent(BaseComponent):
    def __init__(self):
        super().__init__()
        # Optimized parameters from backtesting
        self.BB_PERIOD = 20
        self.BB_STD = 2.0
        self.ATR_PERIOD = 14
        self.KC_PERIOD = 20
        self.KC_MULT = 2.0        # Wider Keltner Channels for better squeeze detection
        self.VOL_LOOKBACK = 100   # Keep standard lookback for volatility calculation
        
    def generate_signal(self, data: Dict[str, np.ndarray]) -> VolatilitySignalResult:
        """Generate volatility-based signals."""
        try:
            # Check if we have enough data points
            close = data['close']
            high = data['high']
            low = data['low']
            
            if len(close) < self.VOL_LOOKBACK:  # Need at least VOL_LOOKBACK points
                return VolatilitySignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    bb_width=0.0,
                    atr_value=0.0,
                    squeeze_strength=0.0
                )
            
            # Calculate Bollinger Bands
            try:
                bb_middle, bb_upper, bb_lower = self._calculate_bollinger_bands(close)
                if len(bb_middle) == 0:
                    raise ValueError("Empty Bollinger Bands")
                bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-9)  # Avoid division by zero
            except Exception:
                return VolatilitySignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    bb_width=0.0,
                    atr_value=0.0,
                    squeeze_strength=0.0
                )
            
            # Calculate ATR
            try:
                atr = self._calculate_atr(high, low, close)
                if len(atr) == 0:
                    raise ValueError("Empty ATR")
            except Exception:
                return VolatilitySignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    bb_width=bb_width[-1],
                    atr_value=0.0,
                    squeeze_strength=0.0
                )
            
            # Calculate Keltner Channels
            try:
                kc_middle, kc_upper, kc_lower = self._calculate_keltner_channels(
                    high, low, close
                )
                if len(kc_middle) == 0:
                    raise ValueError("Empty Keltner Channels")
            except Exception:
                return VolatilitySignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    bb_width=bb_width[-1],
                    atr_value=atr[-1],
                    squeeze_strength=0.0
                )
            
            # Calculate volatility regime
            try:
                returns = np.diff(np.log(close))
                if len(returns) >= self.VOL_LOOKBACK:
                    current_vol = np.std(returns[-20:])
                    historical_vol = np.std(returns[-self.VOL_LOOKBACK:])
                    vol_ratio = current_vol / (historical_vol + 1e-9)  # Avoid division by zero
                else:
                    vol_ratio = 1.0
            except Exception:
                vol_ratio = 1.0
            
            # Calculate squeeze metrics
            squeeze_strength = self._calculate_squeeze(
                bb_upper, bb_lower,
                kc_upper, kc_lower
            )
            
            # Generate normalized signal based on regime
            if vol_ratio > 1.5:  # High volatility
                # More defensive in high volatility
                normalized_signal = self._calculate_high_vol_signal(
                    close[-1], bb_middle[-1], bb_width[-1],
                    atr[-1], squeeze_strength
                )
                confidence = self._calculate_high_vol_confidence(
                    bb_width[-1], vol_ratio, squeeze_strength
                )
            elif vol_ratio < 0.5:  # Low volatility
                # More aggressive in low volatility
                normalized_signal = self._calculate_low_vol_signal(
                    close[-1], bb_middle[-1], bb_width[-1],
                    atr[-1], squeeze_strength
                )
                confidence = self._calculate_low_vol_confidence(
                    bb_width[-1], vol_ratio, squeeze_strength
                )
            else:  # Normal volatility
                normalized_signal = self._calculate_normal_vol_signal(
                    close[-1], bb_middle[-1], bb_width[-1],
                    atr[-1], squeeze_strength
                )
                confidence = self._calculate_normal_vol_confidence(
                    bb_width[-1], vol_ratio, squeeze_strength
                )
                
            return VolatilitySignalResult(
                normalized_signal=normalized_signal,
                confidence=confidence,
                bb_width=bb_width[-1],
                atr_value=atr[-1],
                squeeze_strength=squeeze_strength
            )
            
        except Exception:
            # Return neutral signal on any error
            return VolatilitySignalResult(
                normalized_signal=0.0,
                confidence=0.0,
                bb_width=0.0,
                atr_value=0.0,
                squeeze_strength=0.0
            )
        
    def _calculate_bollinger_bands(
        self,
        close: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        middle = np.array([
            np.mean(close[max(0, i-self.BB_PERIOD+1):i+1])
            for i in range(len(close))
        ])
        
        std = np.array([
            np.std(close[max(0, i-self.BB_PERIOD+1):i+1])
            for i in range(len(close))
        ])
        
        upper = middle + self.BB_STD * std
        lower = middle - self.BB_STD * std
        
        return middle, upper, lower
        
    def _calculate_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> np.ndarray:
        """Calculate Average True Range."""
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Calculate ATR using Wilder's smoothing
        atr = np.zeros_like(close)
        atr[self.ATR_PERIOD] = np.mean(tr[:self.ATR_PERIOD])
        
        for i in range(self.ATR_PERIOD + 1, len(close)):
            atr[i] = (atr[i-1] * (self.ATR_PERIOD - 1) + tr[i-1]) / self.ATR_PERIOD
            
        return atr
        
    def _calculate_keltner_channels(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Keltner Channels."""
        typical_price = (high + low + close) / 3
        
        middle = np.array([
            np.mean(typical_price[max(0, i-self.KC_PERIOD+1):i+1])
            for i in range(len(close))
        ])
        
        atr = self._calculate_atr(high, low, close)
        
        upper = middle + self.KC_MULT * atr
        lower = middle - self.KC_MULT * atr
        
        return middle, upper, lower
        
    def _calculate_squeeze(
        self,
        bb_upper: np.ndarray,
        bb_lower: np.ndarray,
        kc_upper: np.ndarray,
        kc_lower: np.ndarray
    ) -> float:
        """Calculate squeeze strength."""
        # Check if BBs are inside KCs
        bb_width = bb_upper[-1] - bb_lower[-1]
        kc_width = kc_upper[-1] - kc_lower[-1]
        
        if bb_width < kc_width:
            # Calculate how far inside
            squeeze = 1.0 - (bb_width / kc_width)
        else:
            # No squeeze
            squeeze = 0.0
            
        return squeeze
        
    def _calculate_high_vol_signal(
        self,
        close: float,
        bb_middle: float,
        bb_width: float,
        atr: float,
        squeeze: float
    ) -> float:
        """Calculate signal for high volatility regime (long-only)."""
        # Mean reversion focused - buy when price is below mean
        price_deviation = (bb_middle - close) / (atr + 1e-9)
        
        # Generate buy signals when price is below mean
        if price_deviation > 0:
            # Stronger signal when squeeze is present
            if squeeze > 0.5:
                return np.clip(price_deviation * 1.5, 0, 1)
            else:
                return np.clip(price_deviation, 0, 1)
        return 0.0
            
    def _calculate_low_vol_signal(
        self,
        close: float,
        bb_middle: float,
        bb_width: float,
        atr: float,
        squeeze: float
    ) -> float:
        """Calculate signal for low volatility regime (long-only)."""
        # Breakout focused - buy when price is above mean
        price_deviation = (close - bb_middle) / (atr + 1e-9)
        
        # Generate buy signals for upward breakouts
        if price_deviation > 0:
            # Stronger signals when squeeze is present
            if squeeze > 0.5:
                return np.clip(price_deviation * 1.5, 0, 1)
            else:
                return np.clip(price_deviation, 0, 1)
        return 0.0
            
    def _calculate_normal_vol_signal(
        self,
        close: float,
        bb_middle: float,
        bb_width: float,
        atr: float,
        squeeze: float
    ) -> float:
        """Calculate signal for normal volatility regime (long-only)."""
        # Balanced approach
        price_deviation = (close - bb_middle) / (atr + 1e-9)
        
        if squeeze > 0.7:  # Strong squeeze
            # Buy on upward breakouts
            if price_deviation > 0:
                return np.clip(price_deviation * 1.2, 0, 1)
        else:
            # Buy on dips
            if price_deviation < 0:
                return np.clip(-price_deviation * 0.8, 0, 1)
        return 0.0
            
    def _calculate_high_vol_confidence(
        self,
        bb_width: float,
        vol_ratio: float,
        squeeze: float
    ) -> float:
        """Calculate confidence for high volatility regime."""
        # Lower confidence in high volatility
        base_conf = 0.7
        
        # Reduce confidence as volatility increases
        vol_conf = np.clip(2 - vol_ratio, 0, 1)
        
        # Increase confidence if there's a squeeze
        squeeze_conf = squeeze * 0.3
        
        return np.clip(base_conf * vol_conf + squeeze_conf, 0, 1)
        
    def _calculate_low_vol_confidence(
        self,
        bb_width: float,
        vol_ratio: float,
        squeeze: float
    ) -> float:
        """Calculate confidence for low volatility regime."""
        # Higher base confidence in low volatility
        base_conf = 0.8
        
        # Increase confidence as volatility decreases
        vol_conf = np.clip(1 - vol_ratio, 0, 1)
        
        # Increase confidence if there's a squeeze
        squeeze_conf = squeeze * 0.4
        
        return np.clip(base_conf * vol_conf + squeeze_conf, 0, 1)
        
    def _calculate_normal_vol_confidence(
        self,
        bb_width: float,
        vol_ratio: float,
        squeeze: float
    ) -> float:
        """Calculate confidence for normal volatility regime."""
        # Highest base confidence in normal volatility
        base_conf = 0.9
        
        # Reduce confidence as volatility deviates from normal
        vol_conf = 1 - abs(vol_ratio - 1)
        
        # Small boost from squeeze
        squeeze_conf = squeeze * 0.2
        
        return np.clip(base_conf * vol_conf + squeeze_conf, 0, 1)
