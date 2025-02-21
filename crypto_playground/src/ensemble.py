import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from components.trend_component import TrendComponent
from components.momentum_component import MomentumComponent
from components.volatility_component import VolatilityComponent

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class SignalResult:
    combined_signal: float
    confidence: float
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trend_signal: Optional[object] = None
    momentum_signal: Optional[object] = None
    volatility_signal: Optional[object] = None

class EnsembleStrategy:
    def __init__(self):
        # Initialize components with optimized parameters
        self.trend_component = TrendComponent()
        self.momentum_component = MomentumComponent()
        self.volatility_component = VolatilityComponent()
        
        # Best performing parameters from backtesting (64.7% win rate)
        self.MIN_SIGNAL_STRENGTH = 0.1   # Higher threshold for stronger signals
        self.MIN_CONFIDENCE = 0.3        # Moderate confidence requirement
        self.STOP_LOSS_ATR = 2.0        # Wider stop loss for more room
        self.TAKE_PROFIT_ATR = 1.0      # Tighter take profit for faster exits
        
        # Position sizing parameters (3-7% range)
        self.BASE_SIZE = 0.03           # Standard base size
        self.MAX_SIZE = 0.07            # Standard max risk
        
        # Store last analysis result
        self.last_result = None
        
    def analyze(self, data: Dict[str, np.ndarray], current_price: float) -> SignalResult:
        """Generate trading signals using ensemble of components."""
        # Check if we have empty data (outside trading hours)
        if len(data.get('close', [])) == 0:
            return SignalResult(
                combined_signal=0.0,
                confidence=0.0,
                position_size=0.0
            )
            
        # Get component signals
        trend_signal = self.trend_component.generate_signal(data)
        momentum_signal = self.momentum_component.generate_signal(data)
        volatility_signal = self.volatility_component.generate_signal(data)
        
        # Check trend alignment and volume filter
        trend_aligned = data.get('trend_aligned', np.array([False]))[0]  # Use first element since it's a single-element array
        volume_filter = data.get('volume_filter', np.array([False]))[-1] if len(data.get('volume_filter', [])) > 0 else False
        momentum_score = data.get('momentum_score', np.array([0.0]))[-1] if len(data.get('momentum_score', [])) > 0 else 0.0
        stop_mult = data.get('stop_mult', 1.2)  # Default to normal regime
        tp_mult = data.get('tp_mult', 1.8)  # Default to normal regime * 1.5
        
        if not trend_aligned:
            logger.info("Trends not aligned - reducing signal strength")
            trend_signal.normalized_signal *= 0.5
            trend_signal.confidence *= 0.8
            
        # Only apply volume filter when we have actual trades (not quote mid-prices)
        current_volume = data['volume'][-1]
        if current_volume > 0 and not volume_filter:
            logger.info("Volume filter failed - reducing confidence")
            trend_signal.confidence *= 0.7
            momentum_signal.confidence *= 0.7
        elif current_volume == 0:
            logger.info("Volume filter skipped - using quote mid-prices")
            
        # Apply momentum score
        momentum_signal.normalized_signal = np.clip(momentum_signal.normalized_signal + momentum_score * 0.2, -1, 1)
        
        # Volume Analysis (for information only in v1beta3)
        logger.info("\nVolume Analysis:")
        volume = data['volume'][-20:]  # Last 20 periods
        volume_ma = np.mean(volume)
        current_volume = volume[-1]
        
        logger.info(f"Current Volume: {current_volume:.4f}")
        logger.info(f"Volume MA: {volume_ma:.4f}")
        
        # Calculate volume ratio if possible
        volume_ratio = current_volume / volume_ma if volume_ma > 0 else 0.0
        logger.info(f"Volume Ratio: {volume_ratio:.2f}")
        
        # Note: In v1beta3, bars contain quote mid-prices when no trades occur
        # So we don't filter on volume, but use it as additional information
        if current_volume <= 0:
            logger.info("Note: Using quote mid-prices (no trades this period)")
        elif volume_ratio < 0.5:
            logger.info("Note: Low trading activity")
        elif volume_ratio > 2.0:
            logger.info("Note: High trading activity")
            
        logger.info("Volume Filter: Not applicable in v1beta3")
        
        # Volatility Regime Analysis
        logger.info("\nVolatility Analysis:")
        returns = np.diff(np.log(data['close']))
        current_vol = np.std(returns[-20:])
        historical_vol = np.std(returns[-100:])
        vol_ratio = current_vol / historical_vol
        
        logger.info(f"Current/Historical Vol Ratio: {vol_ratio:.2f}")
        
        # Only trade in favorable volatility conditions
        valid_regime = 0.3 <= vol_ratio <= 1.5
        if not valid_regime:
            logger.info("Volatility Filter: FAIL (Outside 0.3-1.5 range)")
            return SignalResult(
                combined_signal=0.0,
                confidence=0.0,
                position_size=0.0,
                trend_signal=trend_signal,
                momentum_signal=momentum_signal,
                volatility_signal=volatility_signal
            )
            
        logger.info("Volatility Filter: PASS")
        
        # Calculate trend conditions
        close_series = pd.Series(data['close'])
        ema10 = close_series.ewm(span=10, adjust=False).mean()
        ema20 = close_series.ewm(span=20, adjust=False).mean()
        ema50 = close_series.ewm(span=50, adjust=False).mean()
        ema100 = close_series.ewm(span=100, adjust=False).mean()
        
        st_trend = (ema10.iloc[-1] / ema20.iloc[-1] - 1) * 100
        lt_trend = (ema50.iloc[-1] / ema100.iloc[-1] - 1) * 100
        trend_aligned = np.sign(st_trend) == np.sign(lt_trend)
        
        # Calculate trend strength
        trend_strength = abs(trend_signal.normalized_signal)
        trend_score = trend_strength / np.max([trend_strength, 0.01])  # Avoid div by zero
        
        # Trend Analysis
        logger.info(f"\nTrend Analysis:")
        logger.info(f"ADX: {trend_signal.adx_strength:.1f} ({'Strong' if trend_signal.adx_strength > 25 else 'Weak'})")
        logger.info(f"Trend Score: {trend_score:.2f}")
        logger.info(f"Short-term Trend: {st_trend:.2f}%")
        logger.info(f"Long-term Trend: {lt_trend:.2f}%")
        logger.info(f"Trends Aligned: {trend_aligned}")
        
        # Strong trend requires high ADX and strong trend score
        strong_trend = trend_signal.adx_strength > 25 and trend_score > 0.6
        logger.info(f"Strong Trend: {'Yes' if strong_trend else 'No'}")
        
        combined_signal = 0.0
        
        # Signal Generation
        logger.info("\nSignal Analysis:")
        
        # Trend following signals (ADX > 25 required)
        if strong_trend and trend_aligned:
            if st_trend > 0:  # Long
                trend_signal_value = (
                    0.5 * trend_signal.dmi_signal +  # DMI direction
                    0.3 * trend_signal.macd_signal +  # MACD momentum
                    0.2 * momentum_signal.normalized_signal  # Overall momentum
                )
                combined_signal = trend_signal_value
                logger.info("Type: Trend Following (Long)")
                logger.info(f"DMI: {trend_signal.dmi_signal:.2f}")
                logger.info(f"MACD: {trend_signal.macd_signal:.2f}")
                logger.info(f"Momentum: {momentum_signal.normalized_signal:.2f}")
                
            elif st_trend < 0:  # Short
                trend_signal_value = (
                    0.5 * trend_signal.dmi_signal +  # DMI direction
                    0.3 * trend_signal.macd_signal +  # MACD momentum
                    0.2 * momentum_signal.normalized_signal  # Overall momentum
                )
                combined_signal = trend_signal_value
                logger.info("Type: Trend Following (Short)")
                logger.info(f"DMI: {trend_signal.dmi_signal:.2f}")
                logger.info(f"MACD: {trend_signal.macd_signal:.2f}")
                logger.info(f"Momentum: {momentum_signal.normalized_signal:.2f}")
                
        # Mean reversion signals (only when ADX <= 25)
        elif trend_signal.adx_strength <= 25:
            logger.info(f"RSI: {momentum_signal.rsi_value:.1f}")
            
            if momentum_signal.rsi_value <= 25:  # Oversold
                rsi_signal = (25 - momentum_signal.rsi_value) / 25  # Scale RSI signal
                mean_rev_signal = (
                    0.5 * rsi_signal +
                    0.3 * momentum_signal.normalized_signal +
                    0.2 * trend_signal.normalized_signal
                )
                combined_signal = mean_rev_signal
                logger.info("Type: Mean Reversion (Long - Oversold)")
                
            elif momentum_signal.rsi_value >= 75:  # Overbought
                rsi_signal = -(momentum_signal.rsi_value - 75) / 25  # Scale RSI signal
                mean_rev_signal = (
                    0.5 * rsi_signal +
                    0.3 * momentum_signal.normalized_signal +
                    0.2 * trend_signal.normalized_signal
                )
                combined_signal = mean_rev_signal
                logger.info("Type: Mean Reversion (Short - Overbought)")
        
        else:
            logger.info("No valid signal pattern detected")
            
        # Calculate confidence
        confidence = np.mean([
            trend_signal.confidence,
            momentum_signal.confidence,
            volatility_signal.confidence
        ])
        
        # Apply signal strength filter
        if abs(combined_signal) < self.MIN_SIGNAL_STRENGTH:
            logger.info(f"Signal strength {abs(combined_signal):.3f} below minimum {self.MIN_SIGNAL_STRENGTH}")
            combined_signal = 0.0
            
        if confidence < self.MIN_CONFIDENCE:
            logger.info(f"Confidence {confidence:.3f} below minimum {self.MIN_CONFIDENCE}")
            combined_signal = 0.0
            
        # Calculate position size
        position_size = self._calculate_position_size(
            combined_signal,
            confidence,
            vol_ratio,
            trend_score
        )
        
        # Calculate stop loss and take profit using dynamic multipliers
        atr = volatility_signal.atr_value
        
        if combined_signal > 0:  # Long position
            stop_loss = current_price - (atr * stop_mult)
            take_profit = current_price + (atr * tp_mult)
        elif combined_signal < 0:  # Short position
            stop_loss = current_price + (atr * stop_mult)
            take_profit = current_price - (atr * tp_mult)
        else:  # No position
            stop_loss = take_profit = None
            
        logger.info(f"\nRisk Parameters:")
        logger.info(f"ATR: {atr:.2f}")
        logger.info(f"Stop Multiplier: {stop_mult:.2f}x")
        logger.info(f"Take Profit Multiplier: {tp_mult:.2f}x")
            
        # Store and return result
        self.last_result = SignalResult(
            combined_signal=combined_signal,
            confidence=confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trend_signal=trend_signal,
            momentum_signal=momentum_signal,
            volatility_signal=volatility_signal
        )
        
        return self.last_result
        
    def _calculate_position_size(
        self,
        signal: float,
        confidence: float,
        vol_ratio: float,
        trend_score: float
    ) -> float:
        """Calculate dynamic position size based on signal quality."""
        if abs(signal) < self.MIN_SIGNAL_STRENGTH or confidence < self.MIN_CONFIDENCE:
            return 0.0
            
        # Base size adjusted for volatility
        base_size = self.BASE_SIZE / np.clip(vol_ratio, 0.5, 2.0)
        
        # Quality multiplier
        signal_quality = abs(signal) * confidence
        
        # Scale based on signal quality
        if signal_quality > 0.8:
            size_cap = 0.07  # Up to 7% for highest quality
        elif signal_quality > 0.6:
            size_cap = 0.05  # Up to 5% for good quality
        else:
            size_cap = 0.03  # Up to 3% for normal trades
            
        # Additional scaling for strong trends
        if trend_score > 0.7:
            base_size *= 1.5
            
        # Calculate final size
        position_size = base_size * confidence
        
        # Apply limits
        return np.clip(position_size, 0.001, size_cap)
        
    def get_trade_params(self) -> Optional[Dict]:
        """Get parameters for trade execution if conditions are met."""
        if not self.last_result:
            return None
            
        if abs(self.last_result.combined_signal) < self.MIN_SIGNAL_STRENGTH:
            return None
            
        if self.last_result.confidence < self.MIN_CONFIDENCE:
            return None
            
        return {
            'signal': self.last_result.combined_signal,
            'confidence': self.last_result.confidence,
            'position_size': self.last_result.position_size,
            'stop_loss': self.last_result.stop_loss,
            'take_profit': self.last_result.take_profit
        }
