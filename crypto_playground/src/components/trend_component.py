import numpy as np
from typing import Dict
from dataclasses import dataclass
from .base_component import BaseComponent, SignalResult

@dataclass
class TrendSignalResult(SignalResult):
    adx_strength: float = 0.0
    dmi_signal: float = 0.0
    macd_signal: float = 0.0
    is_strong_trend: bool = False

class TrendComponent(BaseComponent):
    def __init__(self):
        super().__init__()
        # Optimized parameters from backtesting
        self.ADX_THRESHOLD = 20        # Lower ADX threshold for more signals
        self.DMI_PERIOD = 14           # Period for DI calculations
        self.MACD_FAST = 12            # Fast EMA period
        self.MACD_SLOW = 26            # Slow EMA period
        self.MACD_SIGNAL = 9           # Signal line period
        self.TREND_THRESHOLD = 0.5     # Lower trend threshold for more signals
        
    def generate_signal(self, data: Dict[str, np.ndarray]) -> TrendSignalResult:
        """Generate trend signals using multiple indicators."""
        try:
            # Check if we have enough data points
            high, low, close = data['high'], data['low'], data['close']
            if len(close) < 50:  # Need at least 50 points for longest calculations
                return TrendSignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    adx_strength=0.0,
                    dmi_signal=0.0,
                    macd_signal=0.0,
                    is_strong_trend=False
                )
            
            # Calculate ADX and DMI
            adx, plus_di, minus_di = self._calculate_adx_dmi(high, low, close)
            if len(adx) == 0 or len(plus_di) == 0 or len(minus_di) == 0:
                return TrendSignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    adx_strength=0.0,
                    dmi_signal=0.0,
                    macd_signal=0.0,
                    is_strong_trend=False
                )
            
            # Calculate MACD
            macd, signal, hist = self._calculate_macd(close)
            if len(hist) == 0:
                return TrendSignalResult(
                    normalized_signal=0.0,
                    confidence=0.0,
                    adx_strength=0.0,
                    dmi_signal=0.0,
                    macd_signal=0.0,
                    is_strong_trend=False
                )
            
            # Determine trend strength and direction
            adx_strength = adx[-1]
            is_strong_trend = adx_strength > self.ADX_THRESHOLD
            
            # Calculate DMI signal (-1 to 1)
            dmi_signal = self._calculate_dmi_signal(plus_di[-1], minus_di[-1])
            
            # Calculate MACD signal (-1 to 1)
            macd_signal = self._normalize_macd(hist[-1], hist)
            
            # Calculate trend score
            trend_score = self._calculate_trend_score(close)
            strong_trend = trend_score > self.TREND_THRESHOLD
            
            # Generate normalized signal (long-only)
            if is_strong_trend and strong_trend:
                # Strong trend - only generate signal for uptrends
                if dmi_signal > 0 and macd_signal > 0:
                    normalized_signal = (
                        0.5 * abs(dmi_signal) +
                        0.3 * abs(macd_signal) +
                        0.2 * (1 if close[-1] > close[-20] else 0)  # Medium-term trend
                    )
                else:
                    normalized_signal = 0.0
            else:
                # Weak trend - only generate signal for uptrends
                if macd_signal > 0 and dmi_signal > 0:
                    normalized_signal = (
                        0.4 * abs(macd_signal) +
                        0.3 * abs(dmi_signal) +
                        0.3 * (1 if close[-1] > close[-10] else 0)  # Short-term trend
                    )
                else:
                    normalized_signal = 0.0
                
            # Calculate confidence
            confidence = self._calculate_confidence(
                adx_strength,
                plus_di[-1],
                minus_di[-1],
                hist,
                trend_score
            )
            
            return TrendSignalResult(
                normalized_signal=normalized_signal,
                confidence=confidence,
                adx_strength=adx_strength,
                dmi_signal=dmi_signal,
                macd_signal=macd_signal,
                is_strong_trend=is_strong_trend
            )
            
        except Exception:
            # Return empty signal on any error
            return TrendSignalResult(
                normalized_signal=0.0,
                confidence=0.0,
                adx_strength=0.0,
                dmi_signal=0.0,
                macd_signal=0.0,
                is_strong_trend=False
            )
        
    def _calculate_adx_dmi(self, high: np.ndarray, low: np.ndarray, close: np.ndarray):
        """Calculate ADX and DMI indicators."""
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR and DM
        tr_smooth = self._smooth_data(tr, self.DMI_PERIOD)
        plus_dm_smooth = self._smooth_data(plus_dm, self.DMI_PERIOD)
        minus_dm_smooth = self._smooth_data(minus_dm, self.DMI_PERIOD)
        
        # DI+ and DI-
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # ADX
        # Handle zero division in ADX calculation
        denom = plus_di + minus_di
        dx = np.where(denom > 0, 100 * np.abs(plus_di - minus_di) / denom, 0)  # Fixed comma syntax
        adx = self._smooth_data(dx, self.DMI_PERIOD)
        
        return adx, plus_di, minus_di
        
    def _calculate_macd(self, close: np.ndarray):
        """Calculate MACD indicator."""
        exp1 = self._ema(close, self.MACD_FAST)
        exp2 = self._ema(close, self.MACD_SLOW)
        macd = exp1 - exp2
        signal = self._ema(macd, self.MACD_SIGNAL)
        hist = macd - signal
        return macd, signal, hist
        
    def _calculate_trend_score(self, close: np.ndarray) -> float:
        """Calculate trend strength score."""
        # Check if we have enough data points
        if len(close) < 50:  # Need at least 50 points for longest EMA
            return 0.0
            
        try:
            # Multiple timeframe trend alignment
            ema10 = self._ema(close, 10)
            ema20 = self._ema(close, 20)
            ema50 = self._ema(close, 50)
            
            if len(ema10) < 2 or len(ema20) < 2 or len(ema50) < 2:
                return 0.0
                
            # Calculate trend directions
            st_trend = np.sign(ema10[-1] - ema10[-2])
            mt_trend = np.sign(ema20[-1] - ema20[-2])
            lt_trend = np.sign(ema50[-1] - ema50[-2])
            
            # Calculate trend alignment score
            alignment = (
                (st_trend == mt_trend) * 0.5 +
                (mt_trend == lt_trend) * 0.3 +
                (st_trend == lt_trend) * 0.2
            )
            
            # Calculate trend strength
            if len(close) >= 20:
                returns = np.diff(np.log(close))
                if len(returns) >= 20:
                    volatility = np.std(returns[-20:])
                    if volatility > 0:
                        trend_return = (close[-1] / close[-20] - 1) / volatility
                    else:
                        trend_return = 0
                else:
                    trend_return = 0
            else:
                trend_return = 0
            
            # Combine scores
            return (alignment + np.clip(abs(trend_return) / 2, 0, 1)) / 2
        except Exception:
            return 0.0
        
    def _calculate_dmi_signal(self, plus_di: float, minus_di: float) -> float:
        """Calculate normalized DMI signal (positive for uptrends, zero for downtrends)."""
        di_diff = plus_di - minus_di
        di_sum = plus_di + minus_di
        signal = np.clip(di_diff / (di_sum + 1e-9), -1, 1)  # Avoid division by zero
        return max(0, signal)  # Only return positive values
        
    def _normalize_macd(self, current_hist: float, hist: np.ndarray) -> float:
        """Normalize MACD histogram (positive for bullish, zero for bearish)."""
        hist_std = np.std(hist) + 1e-9  # Avoid division by zero
        signal = np.clip(current_hist / (2 * hist_std), -1, 1)
        return max(0, signal)  # Only return positive values
        
    def _calculate_confidence(
        self,
        adx: float,
        plus_di: float,
        minus_di: float,
        hist: np.ndarray,
        trend_score: float
    ) -> float:
        """Calculate signal confidence."""
        # ADX confidence
        adx_conf = min(adx / 50.0, 1.0)
        
        # DMI confidence
        di_diff = abs(plus_di - minus_di)
        dmi_conf = min(di_diff / 30.0, 1.0)
        
        # MACD confidence
        hist_std = np.std(hist) + 1e-9
        macd_conf = min(abs(hist[-1]) / (2 * hist_std), 1.0)
        
        # Combine confidences with trend score
        confidence = (
            0.3 * adx_conf +
            0.3 * dmi_conf +
            0.2 * macd_conf +
            0.2 * trend_score
        )
        
        return np.clip(confidence, 0, 1)
        
    def _smooth_data(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing."""
        alpha = 1.0 / period
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
        
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        alpha = 2.0 / (period + 1)
        return np.array([
            data[i] if i == 0 else
            data[i] * alpha + ema[i-1] * (1 - alpha)
            for i, ema in enumerate([0] * len(data))
        ])
