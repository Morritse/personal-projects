import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from ensemble import EnsembleStrategy

class HistoricalDataHandler:
    def __init__(
        self,
        symbols: List[str] = ["BTC/USD", "ETH/USD"],
        lookback_days: int = 30
    ):
        self.symbols = symbols
        self.lookback_days = lookback_days
        
    async def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load historical data from CSV files and calculate technical indicators.
        Returns a dictionary of DataFrames with symbols as keys.
        """
        data = {}
        
        for symbol in self.symbols:
            # Load CSV data
            clean_symbol = symbol.replace('/', '')
            filepath = f'src/backtesting/data/{clean_symbol}_5min.csv'
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"No data file found for {symbol}. Please run inspect_bars.py first."
                )
                
            # Read the CSV
            df = pd.read_csv(filepath)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            df['atr'] = self._calculate_atr(df)
            df['rsi'] = self._calculate_rsi(df)
            df['macd'] = self._calculate_macd(df)
            
            # Calculate signals
            df['combined_signal'] = self._calculate_combined_signal(df)
            df['confidence'] = self._calculate_confidence(df)
            df['position_size'] = self._calculate_position_size(df)
            
            # Calculate dynamic stop loss and take profit based on volatility
            volatility = df['close'].pct_change().rolling(window=20).std()
            vol_ratio = volatility / volatility.rolling(window=100).mean()
            
            # Adjust ATR multiplier based on volatility regime
            stop_mult = pd.Series(np.where(vol_ratio > 1.5, 1.0,  # Tighter stops in high vol
                                         np.where(vol_ratio < 0.5, 1.5,  # Wider stops in low vol
                                                1.2)), index=df.index)  # Normal regime
            
            tp_mult = stop_mult * 1.5  # Maintain good risk/reward ratio
            
            df['stop_loss'] = df['close'] - (df['atr'] * stop_mult)
            df['take_profit'] = df['close'] + (df['atr'] * tp_mult)
            
            data[symbol] = df
            
        return data
        
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
        
    def _calculate_macd(self, df: pd.DataFrame) -> pd.Series:
        """Calculate MACD."""
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return (macd - signal).fillna(0)  # Fill NaN with neutral value
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.bfill()  # Backfill NaN values
        
    def _calculate_combined_signal(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trading signal with adaptive regime detection."""
        # Calculate multiple timeframe EMAs
        ema10 = df['close'].ewm(span=10, adjust=False).mean()
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        ema50 = df['close'].ewm(span=50, adjust=False).mean()
        ema100 = df['close'].ewm(span=100, adjust=False).mean()
        
        # Trend strength and direction
        st_trend = (ema10 / ema20 - 1) * 100  # Short-term trend
        lt_trend = (ema50 / ema100 - 1) * 100  # Long-term trend
        
        # Trend alignment
        trend_aligned = pd.Series(np.sign(st_trend) == np.sign(lt_trend), index=df.index)
        
        # Volatility regime
        returns = df['close'].pct_change()
        current_vol = returns.rolling(window=20).std()
        historical_vol = returns.rolling(window=100).std()
        vol_ratio = current_vol / historical_vol
        
        # RSI with dynamic thresholds based on volatility
        rsi = df['rsi']
        rsi_high = pd.Series(np.where(vol_ratio > 1.5, 75, 70), index=df.index)  # Higher threshold in high vol
        rsi_low = pd.Series(np.where(vol_ratio > 1.5, 25, 30), index=df.index)   # Lower threshold in high vol
        
        rsi_signal = pd.Series(
            np.where(rsi > rsi_high, -1.0,  # Overbought
                    np.where(rsi < rsi_low, 1.0,  # Oversold
                            0.0)),  # Neutral
            index=df.index
        )
        
        # MACD momentum with volatility adjustment
        macd = df['macd']
        macd_signal = (macd / (df['close'].rolling(window=20).std() * np.sqrt(vol_ratio))).clip(-1, 1)
        
        # Volume trend
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_trend = df['volume'] / volume_ma
        
        # Base signal on regime
        signal = pd.Series(0.0, index=df.index)
        
        # Only trade in favorable volatility conditions
        valid_regime = (vol_ratio >= 0.3) & (vol_ratio <= 1.5)  # More lenient volatility filter
        
        # Strong trend conditions
        trend_strength = abs(lt_trend)
        trend_score = trend_strength / trend_strength.rolling(window=50).max()  # Shorter lookback
        strong_trend = trend_score > 0.6  # More lenient threshold
        
        # Calculate momentum score with shorter lookback
        momentum = df['close'].pct_change(5)  # 5-period momentum
        mom_std = momentum.rolling(window=20).std()  # 20-period volatility
        momentum_score = (momentum / mom_std).clip(-2, 2)
        
        # Volume filter with more lenient thresholds
        volume_filter = (volume_trend > 1.1) & (volume_trend < 2.5)
        
        # Calculate trend following signals
        trend_signal = pd.Series(0.0, index=df.index)
        trend_signal[valid_regime & trend_aligned & strong_trend & volume_filter] = (
            0.5 * pd.Series(np.sign(lt_trend), index=df.index) +  # Long-term trend direction
            0.3 * momentum_score +                                 # Price momentum
            0.2 * macd_signal                                     # MACD confirmation
        )
        
        # Calculate mean reversion signals (more selective)
        mean_rev_signal = pd.Series(0.0, index=df.index)
        extreme_rsi = (rsi <= 25) | (rsi >= 75)  # More extreme RSI values
        mean_rev_signal[valid_regime & (~strong_trend) & extreme_rsi & volume_filter] = (
            0.5 * rsi_signal +          # RSI signal
            0.3 * macd_signal +         # MACD confirmation
            0.2 * momentum_score        # Price momentum
        )
        
        # Combine signals with time filter
        signal = pd.Series(0.0, index=df.index)
        
        # Only trade during active hours (e.g., avoid low volume periods)
        active_hours = pd.Series(
            df.index.hour.isin(range(8, 20)),  # 8 AM to 8 PM
            index=df.index
        )
        
        signal[active_hours] = trend_signal[active_hours] + mean_rev_signal[active_hours]
        
        return signal.fillna(0).clip(-1, 1)
        
    def _calculate_confidence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate signal confidence with regime-based weighting."""
        # Trend confidence
        ema10 = df['close'].ewm(span=10, adjust=False).mean()
        ema20 = df['close'].ewm(span=20, adjust=False).mean()
        ema50 = df['close'].ewm(span=50, adjust=False).mean()
        ema100 = df['close'].ewm(span=100, adjust=False).mean()
        
        # Short and long-term trend alignment
        st_trend = (ema10 / ema20 - 1) * 100
        lt_trend = (ema50 / ema100 - 1) * 100
        trend_aligned = pd.Series(np.sign(st_trend) == np.sign(lt_trend), index=df.index).astype(float)
        
        # Trend strength confidence
        trend_str = abs(lt_trend)
        trend_conf = trend_str / trend_str.rolling(window=50).max()
        
        # RSI confidence with dynamic thresholds
        rsi = df['rsi']
        returns = df['close'].pct_change()
        vol_ratio = returns.rolling(window=20).std() / returns.rolling(window=100).std()
        
        rsi_high = pd.Series(np.where(vol_ratio > 1.5, 75, 70), index=df.index)
        rsi_low = pd.Series(np.where(vol_ratio > 1.5, 25, 30), index=df.index)
        
        rsi_conf = pd.Series(
            np.where((rsi >= rsi_high) | (rsi <= rsi_low), 1.0,  # High confidence at extremes
                    np.maximum(0, 1 - abs(rsi - 50) / 20)),  # Decreasing confidence towards middle
            index=df.index
        )
        
        # MACD confidence with volatility adjustment
        macd = df['macd']
        price_std = df['close'].rolling(window=20).std()
        macd_conf = (abs(macd) / (price_std * np.sqrt(vol_ratio))).clip(0, 1)
        
        # Volume confidence
        volume = df['volume']
        vol_ma = volume.rolling(window=20).mean()
        vol_ratio = volume / vol_ma
        vol_conf = pd.Series(np.minimum(vol_ratio / 2, 1.0), index=df.index)  # Cap at 2x average volume
        
        # Market regime confidence
        regime_conf = pd.Series(
            np.where((vol_ratio >= 0.3) & (vol_ratio <= 1.5), 1.0, 0.0),  # More lenient volatility filter
            index=df.index
        )
        
        # Combine confidences with regime-dependent weights
        is_trending = trend_str > trend_str.rolling(window=50).mean()
        confidence = pd.Series(0.0, index=df.index)
        
        # Trending regime
        trend_mask = is_trending
        confidence[trend_mask] = (
            0.3 * trend_conf[trend_mask] * trend_aligned[trend_mask] +  # Trend alignment important
            0.2 * macd_conf[trend_mask] +                              # Momentum confirmation
            0.2 * vol_conf[trend_mask] +                               # Volume confirmation
            0.2 * regime_conf[trend_mask] +                            # Market regime
            0.1 * rsi_conf[trend_mask]                                 # RSI less important in trends
        )
        
        # Range regime
        range_mask = ~is_trending
        confidence[range_mask] = (
            0.3 * rsi_conf[range_mask] +                               # RSI more important in ranges
            0.2 * vol_conf[range_mask] +                               # Volume confirmation
            0.2 * regime_conf[range_mask] +                            # Market regime
            0.2 * macd_conf[range_mask] +                              # Momentum still relevant
            0.1 * trend_conf[range_mask]                               # Trend less important in ranges
        )
        
        return confidence.fillna(0).clip(0, 1)
        
    def _calculate_position_size(self, df: pd.DataFrame) -> pd.Series:
        """Calculate dynamic position size based on volatility and confidence."""
        # Calculate volatility-adjusted base size
        volatility = df['close'].pct_change().rolling(window=20).std()
        vol_ratio = volatility / volatility.rolling(window=100).mean()
        
        # Adjust base size inversely to volatility
        base_size = pd.Series(0.02 / vol_ratio.clip(0.5, 2), index=df.index)  # 2% base size
        
        # Scale position size by signal strength and trend alignment
        signal_strength = abs(df['combined_signal'])
        trend_aligned = pd.Series(
            np.sign(df['close'].pct_change(10)) == np.sign(df['close'].pct_change(20)),
            index=df.index
        ).astype(float)
        
        # Increase size for strong signals with trend alignment
        position_scale = np.where(
            (signal_strength > 0.5) & (trend_aligned == 1),
            1.5,  # 50% increase for strong aligned signals
            1.0   # Normal size otherwise
        )
        
        position_size = base_size * df['confidence'] * position_scale
        
        # Dynamic position sizing based on signal quality and confidence
        combined_quality = signal_strength * df['confidence']
        
        # Base size caps with dynamic scaling
        max_size = pd.Series(
            np.where(combined_quality > 0.8, 0.07,  # Up to 7% for highest quality
                    np.where(combined_quality > 0.6, 0.05,  # Up to 5% for good quality
                            0.03)),  # Up to 3% for normal trades
            index=df.index
        )
        
        return position_size.fillna(0).clip(0.001, max_size)
