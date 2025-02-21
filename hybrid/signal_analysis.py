import pandas as pd
import numpy as np
from strategy import IchimokuSuperTrendMACDStrategy
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class SignalAnalyzer:
    def __init__(self):
        self.strategy = IchimokuSuperTrendMACDStrategy()
        
    def calculate_ic(self, predictions, returns):
        """Calculate Information Coefficient (Spearman rank correlation)"""
        # Remove any NaN values
        mask = ~predictions.isna() & ~returns.isna()
        if mask.sum() == 0:
            return np.nan
            
        correlation, _ = spearmanr(predictions[mask], returns[mask])
        return correlation
        
    def analyze_signals(self, symbol, lookback_days=200):
        """Analyze predictive power of each indicator using IC"""
        print(f"\nAnalyzing signals for {symbol}...")
        
        # Load data
        data = self.strategy.get_historical_data(symbol, lookback=lookback_days)
        if data.empty:
            print(f"No data found for {symbol}")
            return
            
        # Calculate indicators
        ichimoku = self.strategy.calculate_ichimoku(data)
        supertrend, direction = self.strategy.calculate_supertrend(data)
        macd, signal, hist = self.strategy.calculate_macd(data)
        
        # Calculate future returns for different horizons
        horizons = [1, 3, 5, 10, 20]
        for horizon in horizons:
            data[f'future_return_{horizon}'] = data['close'].pct_change(horizon).shift(-horizon)
        
        # Calculate combined strategy signal
        signals = pd.Series(index=data.index, dtype=float)
        
        try:
            # Get cloud values
            try:
                cloud_top = ichimoku['cloud'].apply(
                    lambda x: max(x['senkou_span_a'], x['senkou_span_b']), axis=1)
            except:
                # If cloud is not a DataFrame, try direct access
                cloud_top = pd.Series(
                    [max(x['senkou_span_a'], x['senkou_span_b']) 
                     for x in ichimoku['cloud']], 
                    index=data.index
                )
            
            # Calculate signal components as in the strategy
            bullish_ichimoku = pd.Series(True, index=data.index)
            bullish_ichimoku &= data['close'] > cloud_top  # Price above cloud
            bullish_ichimoku &= data['close'] > ichimoku['tenkan_sen']  # Price above conversion
            bullish_ichimoku &= data['close'] > ichimoku['kijun_sen']  # Price above base
            bullish_ichimoku &= ichimoku['tenkan_sen'] > ichimoku['kijun_sen']  # Conversion above base
            
            bullish_supertrend = pd.Series(direction) == 1
            
            bullish_macd = (macd > signal) & (macd > 0)
            
            # Combine signals (-1 to 1 scale)
            signals = (bullish_ichimoku.astype(int) + 
                      bullish_supertrend.astype(int) + 
                      bullish_macd.astype(int) - 1.5) / 1.5
            
            print("\nStrategy Signal Analysis:")
            print("Horizon (days) | IC | Signal/Noise | % Bullish")
            print("-" * 50)
            
            # Calculate metrics for each horizon
            for horizon in horizons:
                future_returns = data[f'future_return_{horizon}']
                ic = self.calculate_ic(signals, future_returns)
                
                # Calculate signal/noise ratio
                signal_changes = (signals != signals.shift(1)).sum()
                signal_noise = len(signals) / signal_changes if signal_changes > 0 else np.inf
                
                # Calculate % of bullish signals
                pct_bullish = (signals > 0).mean()
                
                print(f"{horizon:^13} | {ic:^6.3f} | {signal_noise:^11.1f} | {pct_bullish:^8.1%}")
                
            # Calculate signal transitions
            transitions = pd.crosstab(
                signals.shift(1).round(2), 
                signals.round(2), 
                margins=True
            )
            
            print("\nSignal Transition Matrix:")
            print(transitions)
            
        except Exception as e:
            print(f"Error calculating signals: {str(e)}")

if __name__ == "__main__":
    analyzer = SignalAnalyzer()
    
    # Analyze major tech stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    for symbol in symbols:
        analyzer.analyze_signals(symbol)
