import numpy as np
from typing import Dict
from dataclasses import dataclass
from .base_component import BaseComponent, SignalResult

@dataclass
class MomentumSignalResult(SignalResult):
    rsi_value: float = 0.0
    stoch_rsi: float = 0.0
    cci_value: float = 0.0

class MomentumComponent(BaseComponent):
    def __init__(self):
        super().__init__()
        # Optimized parameters from backtesting
        self.RSI_PERIOD = 14
        self.RSI_HIGH = 70        # Standard overbought
        self.RSI_LOW = 30         # Standard oversold
        self.STOCH_PERIOD = 14
        self.STOCH_K = 3
        self.STOCH_D = 3
        self.CCI_PERIOD = 20
        self.CCI_CONSTANT = 0.015  # Standard CCI calculation constant
        
    def generate_signal(self, data: Dict[str, np.ndarray]) -> MomentumSignalResult:
        """Generate momentum signals using multiple indicators."""
        try:
            # Check if we have enough data points
            close = data['close']
            high = data['high']
            low = data['low']
            
            if len(close) < 100:  # Need at least 100 points for calculations
                return MomentumSignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    rsi_value=50.0,
                    stoch_rsi=0.5,
                    cci_value=0.0
                )
            
            # Calculate RSI
            rsi = self._calculate_rsi(close)
            if len(rsi) == 0:
                return MomentumSignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    rsi_value=50.0,
                    stoch_rsi=0.5,
                    cci_value=0.0
                )
            rsi_value = rsi[-1]
            
            # Calculate Stochastic RSI
            stoch_rsi = self._calculate_stoch_rsi(rsi)
            if len(stoch_rsi) == 0:
                return MomentumSignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    rsi_value=rsi_value,
                    stoch_rsi=0.5,
                    cci_value=0.0
                )
            
            # Calculate CCI
            cci = self._calculate_cci(high, low, close)
            if len(cci) == 0:
                return MomentumSignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    rsi_value=rsi_value,
                    stoch_rsi=stoch_rsi[-1],
                    cci_value=0.0
                )
            cci_value = cci[-1]
            
            # Calculate volatility regime
            returns = np.diff(np.log(close))
            if len(returns) >= 100:
                vol_20 = np.std(returns[-20:])
                vol_100 = np.std(returns[-100:])
                vol_ratio = vol_20 / vol_100 if vol_100 > 0 else 1.0
            else:
                vol_ratio = 1.0
            
            # Adjust thresholds based on volatility
            if vol_ratio > 1.5:  # High volatility
                rsi_high = self.RSI_HIGH + 5
                rsi_low = self.RSI_LOW - 5
            else:  # Normal volatility
                rsi_high = self.RSI_HIGH
                rsi_low = self.RSI_LOW
                
            # Generate normalized signal (long-only)
            rsi_signal = self._calculate_rsi_signal(rsi_value, rsi_high, rsi_low)
            stoch_signal = self._calculate_stoch_signal(stoch_rsi)
            cci_signal = self._calculate_cci_signal(cci_value)
            
            # Weight signals based on market conditions
            if abs(cci_value) > 200:  # Extreme conditions
                # Weight RSI and CCI more heavily
                normalized_signal = max(0,
                    0.4 * rsi_signal +
                    0.2 * stoch_signal +
                    0.4 * cci_signal
                )
            else:
                # More balanced weighting
                normalized_signal = max(0,
                    0.4 * rsi_signal +
                    0.3 * stoch_signal +
                    0.3 * cci_signal
                )
                
            # Calculate confidence
            confidence = self._calculate_confidence(
                rsi_value,
                stoch_rsi,
                cci_value,
                rsi_high,
                rsi_low
            )
            
            return MomentumSignalResult(
                normalized_signal=normalized_signal,
                confidence=confidence,
                rsi_value=rsi_value,
                stoch_rsi=stoch_rsi[-1],
                cci_value=cci_value
            )
            
        except Exception:
            # Return neutral signal on any error
            return MomentumSignalResult(
                normalized_signal=0.0,
                confidence=0.0,
                rsi_value=50.0,
                stoch_rsi=0.5,
                cci_value=0.0
            )
        
    def _calculate_rsi(self, close: np.ndarray) -> np.ndarray:
        """Calculate Relative Strength Index."""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Initialize arrays
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        avg_gain[self.RSI_PERIOD] = np.mean(gain[:self.RSI_PERIOD])
        avg_loss[self.RSI_PERIOD] = np.mean(loss[:self.RSI_PERIOD])
        
        # Calculate subsequent values
        for i in range(self.RSI_PERIOD + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (self.RSI_PERIOD - 1) + gain[i-1]) / self.RSI_PERIOD
            avg_loss[i] = (avg_loss[i-1] * (self.RSI_PERIOD - 1) + loss[i-1]) / self.RSI_PERIOD
            
        # Calculate RS and RSI
        rs = avg_gain[self.RSI_PERIOD:] / (avg_loss[self.RSI_PERIOD:] + 1e-9)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Pad the beginning
        return np.concatenate([np.array([50] * self.RSI_PERIOD), rsi])
        
    def _calculate_stoch_rsi(self, rsi: np.ndarray) -> np.ndarray:
        """Calculate Stochastic RSI."""
        # Calculate StochRSI
        rsi_low = np.array([
            min(rsi[max(0, i-self.STOCH_PERIOD+1):i+1])
            for i in range(len(rsi))
        ])
        rsi_high = np.array([
            max(rsi[max(0, i-self.STOCH_PERIOD+1):i+1])
            for i in range(len(rsi))
        ])
        
        stoch_rsi = (rsi - rsi_low) / (rsi_high - rsi_low + 1e-9)  # Avoid division by zero
        
        # Apply smoothing
        k = self._ema(stoch_rsi, self.STOCH_K)
        d = self._ema(k, self.STOCH_D)
        
        return d
        
    def _calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """Calculate Commodity Channel Index."""
        tp = (high + low + close) / 3
        tp_sma = np.array([
            np.mean(tp[max(0, i-self.CCI_PERIOD+1):i+1])
            for i in range(len(tp))
        ])
        
        mad = np.array([
            np.mean(np.abs(tp[max(0, i-self.CCI_PERIOD+1):i+1] - tp_sma[i]))
            for i in range(len(tp))
        ])
        
        cci = (tp - tp_sma) / (self.CCI_CONSTANT * mad + 1e-9)  # Avoid division by zero
        return cci
        
    def _calculate_rsi_signal(self, rsi: float, high: float, low: float) -> float:
        """Calculate normalized RSI signal (long-only)."""
        if rsi < low:
            # Strong buy on oversold
            return 1.0
        elif rsi < 45:
            # Moderate buy on weakness
            return (45 - rsi) / (45 - low)
        else:
            return 0.0  # No signal for overbought
            
    def _calculate_stoch_signal(self, stoch: np.ndarray) -> float:
        """Calculate normalized Stochastic signal (long-only)."""
        current = stoch[-1]
        if current < 0.2:
            # Strong buy on oversold
            return 1.0
        elif current < 0.4:
            # Moderate buy on weakness
            return (0.4 - current) / 0.2
        else:
            return 0.0  # No signal for overbought
            
    def _calculate_cci_signal(self, cci: float) -> float:
        """Calculate normalized CCI signal (long-only)."""
        if cci < -100:
            # Strong buy on oversold
            return min(-cci / 100, 1.0)
        elif cci < 0:
            # Moderate buy on weakness
            return -cci / 100
        else:
            return 0.0  # No signal for overbought
        
    def _calculate_confidence(
        self,
        rsi: float,
        stoch: np.ndarray,
        cci: float,
        rsi_high: float,
        rsi_low: float
    ) -> float:
        """Calculate signal confidence."""
        # RSI confidence
        if rsi >= rsi_high or rsi <= rsi_low:
            rsi_conf = 1.0  # High confidence at extremes
        else:
            # Decreasing confidence towards middle
            rsi_conf = 1.0 - abs(rsi - 50) / (50 - rsi_low)
            
        # Stochastic confidence
        stoch_current = stoch[-1]
        if stoch_current >= 0.8 or stoch_current <= 0.2:
            stoch_conf = 1.0
        else:
            stoch_conf = 1.0 - abs(stoch_current - 0.5) * 2
            
        # CCI confidence
        cci_conf = min(abs(cci) / 200, 1.0)
        
        # Combine confidences
        confidence = (
            0.4 * rsi_conf +
            0.3 * stoch_conf +
            0.3 * cci_conf
        )
        
        return np.clip(confidence, 0, 1)
        
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        return np.array([
            data[i] if i == 0 else
            data[i] * alpha + ema[i-1] * (1 - alpha)
            for i, ema in enumerate([0] * len(data))
        ])
