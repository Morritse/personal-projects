import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import datetime
import os

def walk_forward_split(df, train_years=3, test_months=6):
    """
    Generator that yields (train_df, test_df) splits in a walk-forward manner.
    Example: train on first 3 years, test on next 6 months, then roll forward.
    """
    df = df.sort_values("date")
    df['date'] = pd.to_datetime(df['date'])  # ensure datetime

    start_date = df['date'].min()
    end_date = df['date'].max()

    # We'll move in increments, each time training on 
    # an expanding window of ~train_years, then test for test_months.
    current_train_start = start_date
    current_train_end = start_date + pd.DateOffset(years=train_years)
    while current_train_end < end_date:
        current_test_end = current_train_end + pd.DateOffset(months=test_months)
        # Limit test_end to not exceed dataset
        if current_test_end > end_date:
            current_test_end = end_date

        train_mask = (df['date'] >= current_train_start) & (df['date'] < current_train_end)
        test_mask = (df['date'] >= current_train_end) & (df['date'] < current_test_end)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) < 50 or len(test_df) < 10:
            break

        yield train_df, test_df

        # Move the window forward
        current_train_end = current_test_end
        if current_train_end >= end_date:
            break

def backtest_5d_hold(test_df, pred_labels):
    """
    Very naive "5-day hold" backtest:
    - If model predicts '1' => go long for the next 5 days
    - We'll assume no overlap (this is simplified)
    - We'll track the sum of returns
    """
    test_df = test_df.reset_index(drop=True)
    total_return = 0.0
    for i in range(len(pred_labels)):
        if pred_labels[i] == 1:
            total_return += test_df.loc[i, 'future_return_5d']
    return total_return

def process_symbol(symbol, feature_cols):
    """Process a single symbol with walk-forward testing"""
    print(f"\nProcessing {symbol}...")
    
    # Load data (using v2 features that include SPY correlation)
    df = pd.read_csv(f"featuresv2_data/{symbol}_with_features_v2.csv", parse_dates=['date'])
    
    # Remove rows without data
    df.dropna(subset=feature_cols + ['target_5d'], inplace=True)

    # Prepare X, y
    X = df[feature_cols]
    y = df['target_5d']

    # Walk-forward testing
    all_test_returns = []
    all_accuracies = []
    split_count = 0

    for train_df, test_df in walk_forward_split(df, train_years=3, test_months=6):
        split_count += 1

        # Train split
        X_train = train_df[feature_cols]
        y_train = train_df['target_5d']
        
        # Test split
        X_test = test_df[feature_cols]
        y_test = test_df['target_5d']

        # Build XGBoost model
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        all_accuracies.append(acc)

        # Simple 5-day hold backtest
        test_return = backtest_5d_hold(test_df, y_pred)
        all_test_returns.append(test_return)

        print(f"[Split {split_count}] Train: {train_df['date'].min().date()} -> {train_df['date'].max().date()}, "
              f"Test: {test_df['date'].min().date()} -> {test_df['date'].max().date()} | "
              f"Accuracy: {acc:.4f}, 5dHold Return: {test_return:.4%}")

        # Feature importance for last split
        if split_count == 1:  # Print feature importance for first split
            importance = model.feature_importances_
            for feat, imp in sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True):
                print(f"{feat}: {imp:.4f}")

    # Summary
    avg_acc = np.mean(all_accuracies) if all_accuracies else 0
    total_return_sum = sum(all_test_returns)
    print(f"\n=== {symbol} Walk-Forward Summary ===")
    print(f"Splits: {split_count}")
    print(f"Average Accuracy: {avg_acc:.4f}")
    print(f"Total 5d Hold Return (sum): {total_return_sum:.4%}")
    
    return {
        'symbol': symbol,
        'avg_accuracy': avg_acc,
        'total_return': total_return_sum,
        'splits': split_count
    }

def main():
    # Define feature columns (including all our features)
    feature_cols = [
        # New features
        'rsi_7', 'rsi_14', 'sma_200', 'corr_20_spy',
        # Original features
        'daily_return', 'log_return', 'sma_20', 'ema_50',
        'bb_upper', 'bb_middle', 'bb_lower',
        'atr_14', 'vol_ratio_20'
    ]
    
    # Process all instruments
    instruments = ["DBC", "DIA", "GLD", "IEF", "IWM", "LQD", "QQQ", "SLV", 
                  "SPY", "TLT", "UNG", "USO", "UUP", "XLE", "XLF"]
    
    results = []
    for symbol in instruments:
        try:
            result = process_symbol(symbol, feature_cols)
            results.append(result)
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    # Overall summary
    print("\n=== Overall Summary ===")
    print("Symbol  | Accuracy | Total Return | Splits")
    print("--------|----------|--------------|-------")
    for r in sorted(results, key=lambda x: x['total_return'], reverse=True):
        print(f"{r['symbol']:<7} | {r['avg_accuracy']:.4f}  | {r['total_return']:.4%}  | {r['splits']}")

if __name__ == "__main__":
    main()
