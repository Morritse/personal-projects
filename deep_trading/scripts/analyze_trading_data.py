import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

def analyze_data_distribution(data, sequences=None):
    """Analyze the distribution of trading data"""
    print("\nData Analysis:")
    print("-" * 50)
    
    # Analyze price movements
    close_prices = data['Close'].values
    daily_returns = np.diff(close_prices) / close_prices[:-1]
    
    print(f"Daily Returns Statistics:")
    print(f"Mean: {np.mean(daily_returns):.4f}")
    print(f"Std: {np.std(daily_returns):.4f}")
    print(f"Min: {np.min(daily_returns):.4f}")
    print(f"Max: {np.max(daily_returns):.4f}")
    print(f"Positive days: {np.mean(daily_returns > 0):.2%}")
    
    # Plot returns distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(daily_returns, bins=50)
    plt.title('Distribution of Daily Returns')
    plt.xlabel('Returns')
    plt.ylabel('Count')
    plt.savefig(os.path.join(RESULTS_DIR, 'returns_distribution.png'))
    plt.close()
    
    # Analyze sequences if provided
    if sequences is not None and len(sequences) > 0:
        seq_array = np.array(sequences)
        print(f"\nSequence Statistics:")
        print(f"Total sequences: {len(sequences)}")
        print(f"Sequence shape: {seq_array.shape}")
        print(f"Contains NaN: {np.any(np.isnan(seq_array))}")
        print(f"Contains Inf: {np.any(np.isinf(seq_array))}")
        
        # Check feature ranges
        print("\nFeature Ranges:")
        for i in range(seq_array.shape[2]):
            feature_data = seq_array[:, :, i]
            print(f"Feature {i}: min={np.min(feature_data):.4f}, max={np.max(feature_data):.4f}, "
                  f"mean={np.mean(feature_data):.4f}, std={np.std(feature_data):.4f}")
            
            # Plot feature distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(feature_data.flatten(), bins=50)
            plt.title(f'Distribution of Feature {i}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.savefig(os.path.join(RESULTS_DIR, f'feature_{i}_distribution.png'))
            plt.close()
    
    return daily_returns

def analyze_trading_data(symbols, start_date='2019-01-01'):
    """Analyze trading data for multiple symbols"""
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date)
        
        # Basic data analysis
        print(f"\nData shape: {data.shape}")
        print("\nMissing values:")
        print(data.isnull().sum())
        
        # Analyze price movements
        daily_returns = analyze_data_distribution(data)
        
        # Analyze autocorrelation
        plt.figure(figsize=(10, 6))
        autocorr = [pd.Series(daily_returns).autocorr(lag=i) for i in range(1, 21)]
        plt.bar(range(1, 21), autocorr)
        plt.title(f'{symbol} Returns Autocorrelation')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.savefig(os.path.join(RESULTS_DIR, f'{symbol}_autocorrelation.png'))
        plt.close()
        
        # Analyze volatility clustering
        rolling_std = pd.Series(daily_returns).rolling(window=20).std()
        plt.figure(figsize=(10, 6))
        rolling_std.plot()
        plt.title(f'{symbol} 20-day Rolling Volatility')
        plt.xlabel('Time')
        plt.ylabel('Volatility')
        plt.savefig(os.path.join(RESULTS_DIR, f'{symbol}_volatility.png'))
        plt.close()
        
        # Save analysis results
        results = {
            'symbol': symbol,
            'mean_return': np.mean(daily_returns),
            'std_return': np.std(daily_returns),
            'skewness': pd.Series(daily_returns).skew(),
            'kurtosis': pd.Series(daily_returns).kurtosis(),
            'positive_days_pct': np.mean(daily_returns > 0)
        }
        
        # Save to CSV
        pd.DataFrame([results]).to_csv(os.path.join(RESULTS_DIR, f'{symbol}_analysis.csv'), index=False)

def main():
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Define symbols to analyze
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'META']
    market_etfs = ['SPY', 'QQQ', 'UVXY', 'SQQQ', 'TLT']
    
    # Analyze all symbols
    print("Analyzing Tech Stocks...")
    analyze_trading_data(tech_stocks)
    
    print("\nAnalyzing Market ETFs...")
    analyze_trading_data(market_etfs)
    
    print("\nAnalysis complete. Results saved in the 'results' directory.")

if __name__ == "__main__":
    main()
