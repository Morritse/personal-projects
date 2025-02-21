from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class SignalBase:
    normalized_signal: float  # -1 to 1
    confidence: float        # 0 to 1
    
class ComponentBase:
    def __init__(self, weight: float = 0.33):
        self.weight = weight
        
    def analyze(self, data: Dict[str, np.ndarray]) -> SignalBase:
        """Base analysis method to be implemented by child classes."""
        raise NotImplementedError("Child classes must implement analyze()")
