import pandas as pd
import numpy as np
from strategy import IchimokuSuperTrendMACDStrategy
from datetime import datetime, timedelta

class IndicatorAnalyzer:
    def __init__(self):
        self.strategy = IchimokuSuperTrendMACDStrategy()
        
    def analyze_indicators(self, symbol, lookback_days=30):
        """Analyze each indicator's predictive accuracy"""
        print(f"\nAnalyzing indicators for {symbol}...")
        
        # Load data
        data = self.strategy.get_historical_data(symbol, lookback=lookback_days)
        if data.empty:
            print(f"No data found for {symbol}")
            return
            
        # Calculate indicators
        ichimoku = self.strategy.calculate_ichimoku(data)
        supertrend, direction = self.strategy.calculate_supertrend(data)
        macd, signal, hist = self.strategy.calculate_macd(data)
        
        # Calculate future returns for different periods
        for period in [1, 3, 5, 10]:
            data[f'future_return_{period}'] = data['close'].pct_change(period).shift(-period)
        
        # Analyze Ichimoku signals
        self.analyze_ichimoku(data, ichimoku)
        
        # Analyze SuperTrend signals
        self.analyze_supertrend(data, direction)
        
        # Analyze MACD signals
        self.analyze_macd(data, macd, signal)
        
    def analyze_ichimoku(self, data, ichimoku):
        """Analyze Ichimoku Cloud indicator accuracy"""
        print("\nIchimoku Analysis:")
        
        # Calculate different Ichimoku signals
        signals = pd.DataFrame(index=data.index)
        
        # Price above/below cloud
        try:
            cloud_top = ichimoku['cloud']['senkou_span_a'].combine(
                ichimoku['cloud']['senkou_span_b'], max)
            cloud_bottom = ichimoku['cloud']['senkou_span_a'].combine(
                ichimoku['cloud']['senkou_span_b'], min)
            signals['above_cloud'] = data['close'] > cloud_top
            signals['below_cloud'] = data['close'] < cloud_bottom
        except:
            print("Could not calculate cloud signals")
            
        # Tenkan/Kijun cross
        signals['tk_cross'] = (
            (ichimoku['tenkan_sen'] > ichimoku['kijun_sen']) & 
            (ichimoku['tenkan_sen'].shift(1) <= ichimoku['kijun_sen'].shift(1))
        )
        
        # Analyze predictive power for each signal type
        for period in [1, 3, 5, 10]:
            future_returns = data[f'future_return_{period}']
            
            # Above cloud accuracy
            if 'above_cloud' in signals.columns:
                accuracy = (signals['above_cloud'] & (future_returns > 0)).mean()
                print(f"{period}d Above Cloud Accuracy: {accuracy:.2%}")
            
            # TK Cross accuracy
            tk_accuracy = (signals['tk_cross'] & (future_returns > 0)).mean()
            print(f"{period}d TK Cross Accuracy: {tk_accuracy:.2%}")
            
    def analyze_supertrend(self, data, direction):
        """Analyze SuperTrend indicator accuracy"""
        print("\nSuperTrend Analysis:")
        
        # Convert direction to DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['bullish'] = pd.Series(direction) == 1
        
        # Analyze predictive power
        for period in [1, 3, 5, 10]:
            future_returns = data[f'future_return_{period}']
            accuracy = (signals['bullish'] & (future_returns > 0)).mean()
            print(f"{period}d Accuracy: {accuracy:.2%}")
            
        # Analyze signal changes
        signal_changes = (signals['bullish'] != signals['bullish'].shift(1)).sum()
        print(f"Signal Changes: {signal_changes}")
        avg_duration = len(signals) / signal_changes if signal_changes > 0 else np.inf
        print(f"Average Signal Duration: {avg_duration:.1f} periods")
        
    def analyze_macd(self, data, macd, signal):
        """Analyze MACD indicator accuracy"""
        print("\nMACD Analysis:")
        
        # Calculate different MACD signals
        signals = pd.DataFrame(index=data.index)
        signals['crossover'] = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        signals['positive'] = macd > 0
        signals['increasing'] = macd > macd.shift(1)
        
        # Analyze predictive power for each signal type
        for period in [1, 3, 5, 10]:
            future_returns = data[f'future_return_{period}']
            
            # Crossover accuracy
            cross_accuracy = (signals['crossover'] & (future_returns > 0)).mean()
            print(f"{period}d Crossover Accuracy: {cross_accuracy:.2%}")
            
            # Positive MACD accuracy
            pos_accuracy = (signals['positive'] & (future_returns > 0)).mean()
            print(f"{period}d Positive MACD Accuracy: {pos_accuracy:.2%}")
            
        # Analyze signal noise
        crossovers = signals['crossover'].sum()
        print(f"Number of Crossovers: {crossovers}")
        avg_duration = len(signals) / crossovers if crossovers > 0 else np.inf
        print(f"Average Signal Duration: {avg_duration:.1f} periods")

if __name__ == "__main__":
    analyzer = IndicatorAnalyzer()
    
    # Analyze major tech stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    for symbol in symbols:
        analyzer.analyze_indicators(symbol)
