import numpy as np
from typing import Dict
from dataclasses import dataclass

@dataclass
class SignalResult:
    """Base class for component signal results."""
    normalized_signal: float = 0.0
    confidence: float = 0.0

class BaseComponent:
    """Base class for trading components."""
    def __init__(self):
        pass
        
    def generate_signal(self, data: Dict[str, np.ndarray]) -> SignalResult:
        """Generate trading signals from data.
        
        Args:
            data: Dictionary containing OHLCV numpy arrays
            
        Returns:
            SignalResult containing normalized signal and confidence
        """
        raise NotImplementedError("Subclasses must implement generate_signal")
