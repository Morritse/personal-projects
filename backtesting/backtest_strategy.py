import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategy import TradingStrategy
from utils.backtester import Backtester
from utils.indicators import BacktestIndicatorCalculator

class BacktestStrategy(TradingStrategy):
    def __init__(
        self,
        symbols: List[str],
        trend_weight: float = 0.2,      # Weight for trend signals
        momentum_weight: float = 0.3,    # Weight for momentum signals
        reversal_weight: float = 0.25,   # Weight for reversal signals
        breakout_threshold: float = 0.2, # Strong signal threshold
        strong_threshold: float = 0.1,   # Medium signal threshold
        weak_threshold: float = 0.03,    # Weak signal threshold
        verbose: bool = True            # Control trade output
    ):
        # Initialize parent class
        super().__init__()
        
        # Store parameters
        self.symbols = symbols
        self.trend_weight = trend_weight
        self.momentum_weight = momentum_weight
        self.reversal_weight = reversal_weight
        self.breakout_threshold = breakout_threshold
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.verbose = verbose
        
        # Configure for backtesting
        self.allow_short_selling = True
        self.min_trade_interval = 0  # No cooldown in backtesting
        
    def backtest(
        self,
        backtester: Backtester,
        signals: Dict[str, Dict[str, Dict[str, float]]],
        timestamp: pd.Timestamp,
        prices: Dict[str, float]
    ):
        """Run backtest using pre-calculated signals."""
        # Process each symbol
        for symbol, timeframe_signals in signals.items():
            # Combine signals and generate decision
            combined_signals = self._combine_timeframe_signals(timeframe_signals)
            ensemble_score = self._calculate_ensemble_score(combined_signals)
            
            # Check current position
            has_position = symbol in backtester.positions
            position_type = backtester.positions[symbol]['trade_type'] if has_position else None
            
            # Generate trading decision
            decision, score = self._generate_decision(ensemble_score, 1 if has_position else 0)
            
            # Get current price
            current_price = prices.get(symbol)
            if current_price is None:
                continue
            
            # Execute trades based on decision and current position
            if decision in ["PRIORITY_BUY", "BUY"]:
                if not has_position:
                    if self.verbose:
                        print(f"\n[TRADE] {timestamp} - {symbol} @ ${current_price:.2f}")
                        print(f"Signal: {ensemble_score:.3f} -> LONG")
                    
                    backtester.enter_position(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price,
                        signal_strength=ensemble_score,
                        trade_type="LONG"
                    )
                elif position_type == "SHORT":
                    if self.verbose:
                        print(f"\n[TRADE] {timestamp} - {symbol} @ ${current_price:.2f}")
                        print(f"Signal: {ensemble_score:.3f} -> REVERSE TO LONG")
                    backtester.exit_position(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price
                    )
                    
                    backtester.enter_position(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price,
                        signal_strength=ensemble_score,
                        trade_type="LONG"
                    )
                elif position_type == "LONG" and ensemble_score < -0.2:  # Only exit on strong reversal
                    # Take profits if signal weakens
                    if self.verbose:
                        print(f"\n[TRADE] {timestamp} - {symbol} @ ${current_price:.2f}")
                        print(f"Signal: {ensemble_score:.3f} -> TAKE PROFIT")
                    backtester.exit_position(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price
                    )
                    
            elif decision in ["PRIORITY_SELL", "SELL"]:
                if has_position and position_type == "LONG":
                    if self.verbose:
                        print(f"\n[TRADE] {timestamp} - {symbol} @ ${current_price:.2f}")
                        print(f"Signal: {ensemble_score:.3f} -> REVERSE TO SHORT")
                    backtester.exit_position(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price
                    )
                    
                    backtester.enter_position(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price,
                        signal_strength=abs(ensemble_score),
                        trade_type="SHORT"
                    )
                elif not has_position:
                    if self.verbose:
                        print(f"\n[TRADE] {timestamp} - {symbol} @ ${current_price:.2f}")
                        print(f"Signal: {ensemble_score:.3f} -> SHORT")
                    
                    backtester.enter_position(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price,
                        signal_strength=abs(ensemble_score),
                        trade_type="SHORT"
                    )
                elif position_type == "SHORT" and ensemble_score > 0.2:  # Only exit on strong reversal
                    # Take profits if signal weakens
                    if self.verbose:
                        print(f"\n[TRADE] {timestamp} - {symbol} @ ${current_price:.2f}")
                        print(f"Signal: {ensemble_score:.3f} -> TAKE PROFIT")
                    backtester.exit_position(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=current_price
                    )
                    
            # Only update equity curve, don't close positions
            backtester.update_equity(timestamp, prices)
            
    def close_all_positions(
        self,
        backtester: Backtester,
        timestamp: pd.Timestamp,
        prices: Dict[str, float]
    ):
        """Close all open positions at the end of the backtest."""
        for symbol, position in list(backtester.positions.items()):
            if symbol in prices:
                if self.verbose:
                    print(f"\n[TRADE] {timestamp} - {symbol} @ ${prices[symbol]:.2f}")
                    print(f"Signal: CLOSE FINAL POSITION")
                backtester.exit_position(
                    symbol=symbol,
                    timestamp=timestamp,
                    price=prices[symbol]
                )
            
    def _combine_timeframe_signals(self, timeframe_signals: dict) -> dict:
        """Use parent's sophisticated signal combination."""
        return super()._combine_timeframe_signals(timeframe_signals)
        
    def _calculate_ensemble_score(self, signals: Dict[str, float]) -> float:
        """Calculate weighted ensemble score with signal amplification."""
        if not signals:
            return 0.0
            
        # Calculate ensemble score using parent's weights
        ensemble_score = super()._calculate_ensemble_score(signals)
        
        # Amplify strong signals
        if abs(ensemble_score) > self.weak_threshold:
            ensemble_score *= 1.5  # Amplify signals that cross threshold
            
        return ensemble_score
        
    def _generate_decision(self, ensemble_score: float, current_position: int) -> tuple:
        """Override parent method to use optimizable thresholds."""
        if ensemble_score >= self.breakout_threshold:
            return "PRIORITY_BUY", ensemble_score
        elif ensemble_score >= self.strong_threshold:
            return "BUY", ensemble_score
        elif ensemble_score <= -self.breakout_threshold:
            return "PRIORITY_SELL", ensemble_score
        elif ensemble_score <= -self.strong_threshold:
            return "SELL", ensemble_score
        elif ensemble_score >= self.weak_threshold and current_position <= 0:
            return "BUY", ensemble_score
        elif ensemble_score <= -self.weak_threshold and current_position >= 0:
            return "SELL", ensemble_score
        else:
            return "HOLD", ensemble_score
