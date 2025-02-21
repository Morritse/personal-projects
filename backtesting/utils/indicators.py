import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from indicators import IndicatorCalculator as BaseIndicatorCalculator
import numpy as np
from config import VERBOSE_INDICATORS

class BacktestIndicatorCalculator(BaseIndicatorCalculator):
    """Extended indicator calculator with debug logging for backtesting."""
    
    def _calculate_sma(self, data, period: int = 20) -> float:
        """Calculate Simple Moving Average with debug logging."""
        close = data['close'].values
        if len(close) < period:
            if VERBOSE_INDICATORS:
                print(f"Not enough data for SMA: need {period}, have {len(close)}")
            return None
            
        sma = super()._calculate_sma(data, period)
        if sma is not None and VERBOSE_INDICATORS:
            print(f"SMA calculation successful:")
            print(f"- Data points: {len(close)}")
            print(f"- Period: {period}")
            print(f"- Signal: {sma:.3f}")
            
        return sma
        
    def _calculate_ema(self, data, period: int = 20) -> float:
        """Calculate EMA with debug logging."""
        close = data['close'].values
        if len(close) < period:
            if VERBOSE_INDICATORS:
                print(f"Not enough data for EMA: need {period}, have {len(close)}")
            return None
            
        ema = super()._calculate_ema(data, period)
        if ema is not None and VERBOSE_INDICATORS:
            print(f"EMA calculation successful:")
            print(f"- Data points: {len(close)}")
            print(f"- Period: {period}")
            print(f"- Signal: {ema:.3f}")
            
        return ema
        
    def _calculate_adx(self, data, period: int = 14) -> float:
        """Calculate ADX with debug logging."""
        close = data['close'].values
        if len(close) < period * 2:  # ADX needs 2x period
            if VERBOSE_INDICATORS:
                print(f"Not enough data for ADX: need {period*2}, have {len(close)}")
            return None
            
        adx = super()._calculate_adx(data, period)
        if adx is not None and VERBOSE_INDICATORS:
            print(f"ADX calculation successful:")
            print(f"- Data points: {len(close)}")
            print(f"- Period: {period}")
            print(f"- Signal: {adx:.3f}")
            
        return adx
        
    def _calculate_trix(self, data, period: int = 20) -> float:
        """Calculate TRIX with debug logging."""
        close = data['close'].values
        if len(close) < period * 3:  # TRIX needs 3x period
            if VERBOSE_INDICATORS:
                print(f"Not enough data for TRIX: need {period*3}, have {len(close)}")
            return None
            
        trix = super()._calculate_trix(data, period)
        if trix is not None and VERBOSE_INDICATORS:
            print(f"TRIX calculation successful:")
            print(f"- Data points: {len(close)}")
            print(f"- Period: {period}")
            print(f"- Signal: {trix:.3f}")
            
        return trix
        
    def _calculate_macd(self, data, period: int = None) -> float:
        """Calculate MACD with debug logging."""
        close = data['close'].values
        if len(close) < 35:  # Need 26 + 9 bars
            if VERBOSE_INDICATORS:
                print(f"Not enough data for MACD: need 35, have {len(close)}")
            return None
            
        macd = super()._calculate_macd(data, period)
        if macd is not None and VERBOSE_INDICATORS:
            print(f"MACD calculation successful:")
            print(f"- Data points: {len(close)}")
            print(f"- Signal: {macd:.3f}")
            
        return macd
