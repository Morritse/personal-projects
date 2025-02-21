import numpy as np
import talib
from typing import Dict, Optional
from config import (
    TIMEFRAME_INDICATORS,
    VERBOSE_INDICATORS
)

class IndicatorCalculator:
    def __init__(self):
        # Organize indicators by their strategic purpose
        self.indicators = {
            # Primary Trend Indicators (core trend direction)
            'SMA': self._calculate_sma,      # Simple Moving Average - base trend
            'EMA': self._calculate_ema,      # Exponential Moving Average - weighted trend
            'MACD': self._calculate_macd,    # Moving Average Convergence/Divergence - trend momentum
            'ADX': self._calculate_adx,      # Average Directional Index - trend strength
            'TRIX': self._calculate_trix,    # Triple Exponential Average - filtered trend
            
            # Price Action Indicators (market structure)
            'BBANDS': self._calculate_bbands,  # Bollinger Bands - volatility and price channels
            'SAR': self._calculate_sar,        # Parabolic SAR - potential reversal points
            'ROC': self._calculate_roc,        # Rate of Change - price momentum
            
            # Momentum Oscillators (overbought/oversold)
            'RSI': self._calculate_rsi,        # Relative Strength Index - momentum and reversals
            'STOCH': self._calculate_stoch,    # Stochastic - momentum and price position
            'CCI': self._calculate_cci,        # Commodity Channel Index - deviation from mean
            'MFI': self._calculate_mfi,        # Money Flow Index - volume-weighted RSI
            
            # Volume Analysis (confirmation)
            'OBV': self._calculate_obv,        # On Balance Volume - cumulative volume pressure
            'AD': self._calculate_ad,          # Accumulation/Distribution - volume price agreement
            'ADOSC': self._calculate_adosc,    # Chaikin Oscillator - volume momentum
            
            # Volatility Measures (risk management)
            'ATR': self._calculate_atr,        # Average True Range - volatility measurement
            'NATR': self._calculate_natr,      # Normalized ATR - percentage volatility
            'STDDEV': self._calculate_stddev   # Standard Deviation - price dispersion
        }

    def _normalize_signal(self, value: float, min_val: float = -100, max_val: float = 100) -> float:
        """Normalize signal to [-1, 1] range."""
        if VERBOSE_INDICATORS:
            print(f"  Normalizing: value={value:.3f}, min={min_val:.3f}, max={max_val:.3f}")
        
        if min_val == max_val:
            if VERBOSE_INDICATORS:
                print("  Range is zero, returning neutral signal")
            return 0.0
            
        if np.isnan(value) or np.isnan(min_val) or np.isnan(max_val):
            if VERBOSE_INDICATORS:
                print("  NaN value detected, returning neutral signal")
            return 0.0
            
        try:
            normalized = 2 * (value - min_val) / (max_val - min_val) - 1
            clipped = max(min(normalized, 1.0), -1.0)
            if VERBOSE_INDICATORS:
                print(f"  Normalized: {normalized:.3f}, Clipped: {clipped:.3f}")
            return clipped
        except (ZeroDivisionError, RuntimeWarning) as e:
            if VERBOSE_INDICATORS:
                print(f"  Error during normalization: {str(e)}")
            return 0.0

    def _get_latest_value(self, arr: np.ndarray) -> Optional[float]:
        """Safely get the latest non-NaN value from an array."""
        if arr is None:
            if VERBOSE_INDICATORS:
                print("  Array is None")
            return None
        if len(arr) == 0:
            if VERBOSE_INDICATORS:
                print("  Array is empty")
            return None
            
        # Get last non-NaN value
        valid_values = arr[~np.isnan(arr)]
        if len(valid_values) == 0:
            if VERBOSE_INDICATORS:
                print("  No valid (non-NaN) values found")
            return None
            
        value = float(valid_values[-1])
        if VERBOSE_INDICATORS:
            print(f"  Latest value: {value:.3f}")
        return value

    def get_indicator_signals(self, data, timeframe: str = 'short') -> Dict[str, float]:
        """Calculate timeframe-specific indicators and return their signals."""
        if data.empty:
            return {}

        # Get indicators specific to this timeframe
        timeframe_indicators = TIMEFRAME_INDICATORS.get(timeframe, {})
        
        signals = {}
        if VERBOSE_INDICATORS:
            print(f"\nCalculating indicator signals for {timeframe} timeframe...")
            
        for name, period in timeframe_indicators.items():
            try:
                func = self.indicators.get(name)
                if func:
                    signal = func(data, period=period)
                    if signal is not None:
                        signals[name] = signal
                        if VERBOSE_INDICATORS:
                            print(f"{name}: {signal:.3f}")
                    elif VERBOSE_INDICATORS:
                        print(f"{name}: None (insufficient data)")
            except Exception as e:
                if VERBOSE_INDICATORS:
                    print(f"Error calculating {name}: {str(e)}")
        
        if VERBOSE_INDICATORS:
            print(f"\nCalculated {len(signals)} valid signals")

        return signals

    def _calculate_sma(self, data, period: int = 20) -> Optional[float]:
        close = data['close'].values
        sma = talib.SMA(close, timeperiod=period)
        latest = self._get_latest_value(sma)
        if latest is None:
            return None
        return self._normalize_signal(latest, min(close), max(close))

    def _calculate_ema(self, data, period: int = 20) -> Optional[float]:
        close = data['close'].values
        ema = talib.EMA(close, timeperiod=period)
        latest = self._get_latest_value(ema)
        if latest is None:
            return None
        return self._normalize_signal(latest, min(close), max(close))

    def _calculate_macd(self, data, period: int = None) -> Optional[float]:
        close = data['close'].values
        if len(close) < 26:  # Need at least 26 bars for MACD
            return None
        macd, signal, hist = talib.MACD(close, 
            fastperiod=12, 
            slowperiod=26, 
            signalperiod=9)
        latest_hist = self._get_latest_value(hist)
        if latest_hist is None:
            return None
        return self._normalize_signal(latest_hist, -10, 10)

    def _calculate_bbands(self, data, period: int = 20) -> Optional[float]:
        close = data['close'].values
        upper, middle, lower = talib.BBANDS(close, timeperiod=period)
        latest_upper = self._get_latest_value(upper)
        latest_lower = self._get_latest_value(lower)
        latest_close = close[-1]
        if None in (latest_upper, latest_lower, latest_close):
            return None
        band_width = latest_upper - latest_lower
        if band_width == 0:
            return 0
        position = (latest_close - latest_lower) / band_width
        return 2 * position - 1

    def _calculate_rsi(self, data, period: int = 14) -> Optional[float]:
        close = data['close'].values
        rsi = talib.RSI(close, timeperiod=period)
        latest = self._get_latest_value(rsi)
        if latest is None:
            return None
        return self._normalize_signal(latest, 0, 100)

    def _calculate_stoch(self, data, period: int = None) -> Optional[float]:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        slowk, slowd = talib.STOCH(high, low, close,
            fastk_period=14,
            slowk_period=3,
            slowd_period=3)
        latest_k = self._get_latest_value(slowk)
        if latest_k is None:
            return None
        return self._normalize_signal(latest_k, 0, 100)

    def _calculate_cci(self, data, period: int = 14) -> Optional[float]:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        cci = talib.CCI(high, low, close, timeperiod=period)
        latest = self._get_latest_value(cci)
        if latest is None:
            return None
        return self._normalize_signal(latest, -200, 200)

    def _calculate_mfi(self, data, period: int = 14) -> Optional[float]:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        mfi = talib.MFI(high, low, close, volume, timeperiod=period)
        latest = self._get_latest_value(mfi)
        if latest is None:
            return None
        return self._normalize_signal(latest, 0, 100)

    def _calculate_ad(self, data, period: int = None) -> Optional[float]:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        ad = talib.AD(high, low, close, volume)
        latest = self._get_latest_value(ad)
        if latest is None:
            return None
        return self._normalize_signal(latest, min(ad), max(ad))

    def _calculate_obv(self, data, period: int = None) -> Optional[float]:
        close = data['close'].values
        volume = data['volume'].values
        obv = talib.OBV(close, volume)
        latest = self._get_latest_value(obv)
        if latest is None:
            return None
        return self._normalize_signal(latest, min(obv), max(obv))

    def _calculate_adosc(self, data, period: int = None) -> Optional[float]:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        volume = data['volume'].values
        adosc = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10)
        latest = self._get_latest_value(adosc)
        if latest is None:
            return None
        window = min(len(adosc), 20)
        recent_adosc = adosc[-window:]
        min_val = np.nanmin(recent_adosc)
        max_val = np.nanmax(recent_adosc)
        if np.isnan(min_val) or np.isnan(max_val):
            return 0.0
        return self._normalize_signal(latest, min_val, max_val)

    def _calculate_atr(self, data, period: int = 14) -> Optional[float]:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        atr = talib.ATR(high, low, close, timeperiod=period)
        latest = self._get_latest_value(atr)
        if latest is None:
            return None
        window = min(len(atr), 20)
        recent_atr = atr[-window:]
        max_atr = np.nanmax(recent_atr)
        if np.isnan(max_atr) or max_atr == 0:
            return 0.0
        return self._normalize_signal(latest, 0, max_atr)

    def _calculate_natr(self, data, period: int = 14) -> Optional[float]:
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        natr = talib.NATR(high, low, close, timeperiod=period)
        latest = self._get_latest_value(natr)
        if latest is None:
            return None
        return self._normalize_signal(latest, 0, 100)

    def _calculate_adx(self, data, period: int = 14) -> Optional[float]:
        """Average Directional Index - trend strength indicator."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        adx = talib.ADX(high, low, close, timeperiod=period)
        latest = self._get_latest_value(adx)
        if latest is None:
            return None
        return self._normalize_signal(latest, 0, 100)  # ADX ranges from 0 to 100

    def _calculate_trix(self, data, period: int = 20) -> Optional[float]:
        close = data['close'].values
        trix = talib.TRIX(close, timeperiod=period)
        latest = self._get_latest_value(trix)
        if latest is None:
            return None
        return self._normalize_signal(latest, -1, 1)

    def _calculate_sar(self, data, period: int = None) -> Optional[float]:
        high = data['high'].values
        low = data['low'].values
        sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
        latest = self._get_latest_value(sar)
        latest_close = data['close'].values[-1]
        if latest is None:
            return None
        return 1.0 if latest_close > latest else -1.0

    def _calculate_roc(self, data, period: int = 10) -> Optional[float]:
        close = data['close'].values
        roc = talib.ROC(close, timeperiod=period)
        latest = self._get_latest_value(roc)
        if latest is None:
            return None
        return self._normalize_signal(latest, -20, 20)

    def _calculate_stddev(self, data, period: int = 20) -> Optional[float]:
        close = data['close'].values
        stddev = talib.STDDEV(close, timeperiod=period, nbdev=1)
        latest = self._get_latest_value(stddev)
        if latest is None:
            return None
        window = min(len(stddev), 20)
        recent_stddev = stddev[-window:]
        max_dev = np.nanmax(recent_stddev)
        if np.isnan(max_dev) or max_dev == 0:
            return 0.0
        return self._normalize_signal(latest, 0, max_dev)
