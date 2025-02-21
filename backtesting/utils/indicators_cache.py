import pandas as pd
import numpy as np
from typing import Dict, Optional
from config import TIMEFRAME_INDICATORS
from utils.indicators import BacktestIndicatorCalculator

class IndicatorCache:
    """Cache for pre-calculated indicators to avoid recalculation."""
    
    def __init__(self):
        self.calculator = BacktestIndicatorCalculator()
        self.cache = {}
        
    def precalculate_indicators(
        self, 
        data: Dict[str, Dict[str, pd.DataFrame]], 
        progress_callback: Optional[callable] = None
    ) -> Dict:
        """Pre-calculate all indicators for the entire dataset."""
        cached_data = {}
        
        for symbol, timeframe_data in data.items():
            symbol_cache = {}
            for timeframe, df in timeframe_data.items():
                # Calculate indicators for this timeframe
                indicators = {}
                for name, period in TIMEFRAME_INDICATORS.get(timeframe, {}).items():
                    try:
                        # Calculate indicator for entire dataset
                        func = self.calculator.indicators.get(name)
                        if func:
                            signal = func(df, period=period)
                            if signal is not None:
                                indicators[name] = signal
                    except Exception as e:
                        print(f"Error calculating {name} for {symbol} {timeframe}: {str(e)}")
                    
                    # Update progress if callback provided
                    if progress_callback:
                        progress_callback()
                        
                symbol_cache[timeframe] = indicators
            cached_data[symbol] = symbol_cache
            
        self.cache = cached_data
        return cached_data
        
    def get_signals_at_timestamp(self, timestamp: pd.Timestamp) -> Dict:
        """Get all signals at a specific timestamp."""
        signals = {}
        
        for symbol, timeframe_data in self.cache.items():
            timeframe_signals = {}
            for timeframe, indicators in timeframe_data.items():
                # Get all indicator values for this timeframe
                timeframe_signals[timeframe] = indicators
            signals[symbol] = timeframe_signals
            
        return signals
