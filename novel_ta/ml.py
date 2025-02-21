import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

def train_evaluate_symbol(symbol, input_path, split_date="2022-01-01"):
    """
    Train and evaluate a model for a single symbol
    """
    print(f"\nProcessing {symbol}...")
    
    # 1) Load data
    df = pd.read_csv(input_path, parse_dates=['date'])
    df.sort_values("date", inplace=True)
    
    # 2) Create label
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    df['target'] = (df['future_return'] > 0).astype(int)
    
    # 3) Drop NaNs
    df.dropna(inplace=True)
    
    # 4) Define features
    features = ['daily_return', 'log_return', 'sma_20', 'ema_50', 
                'bb_upper', 'bb_middle', 'bb_lower', 
                'rsi_14', 'atr_14', 'vol_ratio_20']
    X = df[features]
    y = df['target']
    
    # 5) Train/test split
    X_train = X[df['date'] < split_date]
    y_train = y[df['date'] < split_date]
    X_test = X[df['date'] >= split_date]
    y_test = y[df['date'] >= split_date]
    
    # 6) Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    # 7) Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 8) Get feature importances
    importances = dict(zip(features, model.feature_importances_))
    
    return {
        'symbol': symbol,
        'accuracy': acc,
        'feature_importances': importances,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'classification_report': classification_report(y_test, y_pred)
    }

if __name__ == "__main__":
    # List of all instruments
    instruments = ["DBC", "DIA", "GLD", "IEF", "IWM", "LQD", "QQQ", "SLV", 
                  "SPY", "TLT", "UNG", "USO", "UUP", "XLE", "XLF"]
    
    # Store results for all symbols
    results = []
    
    for symbol in instruments:
        input_path = f"5_year_with_indicators/{symbol}_with_features.csv"
        
        # Skip if processed file doesn't exist
        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping {symbol}")
            continue
            
        try:
            result = train_evaluate_symbol(symbol, input_path)
            results.append(result)
            
            # Print individual symbol results
            print(f"\nResults for {symbol}:")
            print(f"Accuracy: {result['accuracy']:.4f}")
            print("\nTop 3 important features:")
            sorted_features = sorted(result['feature_importances'].items(), 
                                  key=lambda x: x[1], reverse=True)[:3]
            for feature, importance in sorted_features:
                print(f"{feature}: {importance:.4f}")
            print("\nClassification Report:")
            print(result['classification_report'])
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    # Print summary statistics
    if results:
        accuracies = [r['accuracy'] for r in results]
        print("\n=== Overall Summary ===")
        print(f"Average Accuracy: {np.mean(accuracies):.4f}")
        print(f"Best Performing: {results[np.argmax(accuracies)]['symbol']} "
              f"({max(accuracies):.4f})")
        print(f"Worst Performing: {results[np.argmin(accuracies)]['symbol']} "
              f"({min(accuracies):.4f})")
