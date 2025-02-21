from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from components.trend_component import TrendComponent, TrendSignal
from components.momentum_component import MomentumComponent, MomentumSignal
from components.volatility_component import VolatilityComponent, VolatilitySignal

@dataclass
class EnsembleSignal:
    trend_signal: TrendSignal
    momentum_signal: MomentumSignal
    volatility_signal: VolatilitySignal
    combined_signal: float
    confidence: float
    metadata: Dict
    position_size: float
    stop_loss: Optional[float]
    take_profit: Optional[float]

class EnsembleStrategy:
    """
    Combines signals from trend, momentum, and volatility components.
    """
    def __init__(
        self,
        trend_weight: float = 0.4,
        momentum_weight: float = 0.3,
        volatility_weight: float = 0.3,
        base_position_size: float = 5000.0,
        max_position_size: float = 10000.0,
        stop_loss_atr_factor: float = 1.5,
        take_profit_atr_factor: float = 3.0,
        min_confidence: float = 0.5,
        squeeze_boost: float = 1.5,
        trend_confirmation: bool = False
    ):
        # Initialize components
        self.trend = TrendComponent(weight=trend_weight)
        self.momentum = MomentumComponent(weight=momentum_weight)
        self.volatility = VolatilityComponent(weight=volatility_weight)
        
        # Store parameters
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.stop_loss_atr_factor = stop_loss_atr_factor
        self.take_profit_atr_factor = take_profit_atr_factor
        self.min_confidence = min_confidence
        self.squeeze_boost = squeeze_boost
        self.trend_confirmation = trend_confirmation
        
        # Track last signal
        self.last_signal: Optional[EnsembleSignal] = None

    def analyze(self, data: Dict[str, np.ndarray], current_price: float) -> EnsembleSignal:
        """Analyze market data using all components and generate combined signal."""
        # Get component signals
        trend_signal = self.trend.analyze(data)
        momentum_signal = self.momentum.analyze(data)
        volatility_signal = self.volatility.analyze(data)
        
        # Calculate base signal with dynamic weights
        weights = {
            'trend': self.trend.weight,
            'momentum': self.momentum.weight,
            'volatility': self.volatility.weight
        }
        
        # Adjust weights based on conditions
        if trend_signal.is_strong_trend:
            weights['trend'] *= 1.5
            
        if momentum_signal.is_overbought or momentum_signal.is_oversold:
            weights['momentum'] *= 1.3
            
        if volatility_signal.is_squeeze:
            weights['volatility'] *= self.squeeze_boost
            
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculate base signal
        base_signal = (
            trend_signal.normalized_signal * weights['trend'] +
            momentum_signal.normalized_signal * weights['momentum'] +
            volatility_signal.normalized_signal * weights['volatility']
        )
        
        # Apply squeeze boost if active
        if volatility_signal.is_squeeze:
            base_signal *= self.squeeze_boost
        
        # Apply trend confirmation if required
        if self.trend_confirmation and not trend_signal.is_strong_trend:
            base_signal *= 0.5
        
        # Normalize final signal
        combined_signal = np.clip(base_signal, -1.0, 1.0)
        
        # Calculate enhanced confidence
        base_confidence = self._calculate_confidence(
            trend_signal,
            momentum_signal,
            volatility_signal
        )
        
        # Boost confidence during strong trends
        if trend_signal.is_strong_trend:
            confidence = min(base_confidence * 1.5, 1.0)
        else:
            confidence = base_confidence
        
        # Dynamic position sizing based on volatility and confidence
        position_size, stop_loss, take_profit = self._calculate_position_params(
            combined_signal,
            confidence,
            current_price,
            volatility_signal.atr_value
        )
        
        # Create ensemble signal
        signal = EnsembleSignal(
            trend_signal=trend_signal,
            momentum_signal=momentum_signal,
            volatility_signal=volatility_signal,
            combined_signal=combined_signal,
            confidence=confidence,
            metadata={
                'trend_confidence': trend_signal.confidence,
                'momentum_confidence': momentum_signal.confidence,
                'volatility_confidence': volatility_signal.confidence,
                'is_strong_trend': trend_signal.is_strong_trend,
                'is_squeeze': volatility_signal.is_squeeze,
                'breakout_direction': volatility_signal.breakout_direction,
                'weights': weights
            },
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.last_signal = signal
        return signal

    def _calculate_confidence(
        self,
        trend: TrendSignal,
        momentum: MomentumSignal,
        volatility: VolatilitySignal
    ) -> float:
        """Calculate overall confidence based on component signals."""
        # Base confidence on signal agreement
        signal_agreement = 1.0 - (
            abs(trend.normalized_signal - momentum.normalized_signal) +
            abs(momentum.normalized_signal - volatility.normalized_signal) +
            abs(volatility.normalized_signal - trend.normalized_signal)
        ) / 6.0  # Normalize to 0-1
        
        # Weight individual confidences
        weighted_confidence = (
            trend.confidence * self.trend.weight +
            momentum.confidence * self.momentum.weight +
            volatility.confidence * self.volatility.weight
        )
        
        # Calculate signal strength
        signal_strength = (
            abs(trend.normalized_signal) * self.trend.weight +
            abs(momentum.normalized_signal) * self.momentum.weight +
            abs(volatility.normalized_signal) * self.volatility.weight
        )
        
        # Combine metrics
        confidence = (
            signal_agreement * 0.3 +
            weighted_confidence * 0.3 +
            signal_strength * 0.4
        )
        
        # Boost confidence if components agree
        if all(c > 0.4 for c in [trend.confidence, momentum.confidence, volatility.confidence]):
            confidence = min(1.0, confidence * 1.5)
            
        # Ensure confidence is in 0-1 range
        return max(0.0, min(1.0, confidence))

    def _calculate_position_params(
        self,
        signal: float,
        confidence: float,
        current_price: float,
        atr: float
    ) -> tuple[float, Optional[float], Optional[float]]:
        """Calculate position size and risk levels."""
        if confidence < self.min_confidence:
            return 0.0, None, None
            
        # Scale position size by signal strength and confidence
        signal_scale = abs(signal)
        position_scale = min(signal_scale * confidence, 1.0)
        position_size = self.base_position_size * position_scale
        
        # Cap at maximum size
        position_size = min(position_size, self.max_position_size)
        
        # Calculate actual position in units
        units = position_size / current_price
        
        # Calculate stop loss and take profit levels
        if signal > 0:  # Long position
            stop_loss = current_price - (atr * self.stop_loss_atr_factor)
            take_profit = current_price + (atr * self.take_profit_atr_factor)
        else:  # Short position
            stop_loss = current_price + (atr * self.stop_loss_atr_factor)
            take_profit = current_price - (atr * self.take_profit_atr_factor)
            
        return units, stop_loss, take_profit

    def should_trade(self) -> bool:
        """Determine if conditions are right for trading."""
        if not self.last_signal:
            return False
            
        # Check confidence threshold
        if self.last_signal.confidence < self.min_confidence:
            return False
            
        # Check for signals
        if abs(self.last_signal.combined_signal) < 0.05:
            return False
            
        return True

    def get_trade_params(self) -> Optional[Dict]:
        """Get trading parameters if conditions are met."""
        if not self.should_trade() or not self.last_signal:
            return None
            
        return {
            'signal': self.last_signal.combined_signal,
            'confidence': self.last_signal.confidence,
            'position_size': self.last_signal.position_size,
            'stop_loss': self.last_signal.stop_loss,
            'take_profit': self.last_signal.take_profit,
            'metadata': self.last_signal.metadata
        }
