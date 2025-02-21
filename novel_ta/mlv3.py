import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

################################################################################
# 1) Combine All Symbols into One DataFrame
################################################################################

def load_and_combine_data(symbols, feature_cols):
    """
    For each symbol, load its CSV (with features),
    add a 'symbol' column, and concatenate into one big DataFrame.
    
    Assumes each CSV has at least: [date, target_5d, future_return_5d, ... features ...].
    """
    dfs = []
    for sym in symbols:
        filepath = f"featuresv2_data/{sym}_with_features_v2.csv"
        if not os.path.exists(filepath):
            print(f"[WARN] {filepath} not found. Skipping {sym}.")
            continue
        
        df_sym = pd.read_csv(filepath, parse_dates=['date'])
        df_sym['symbol'] = sym
        
        # Keep only columns of interest plus 'symbol'
        needed = ['date','symbol','target_5d','future_return_5d'] + feature_cols
        # If some columns are missing, skip them but warn
        missing_cols = set(needed) - set(df_sym.columns)
        if missing_cols:
            print(f"[WARN] {sym} missing columns {missing_cols}, skipping those.")
        use_cols = [c for c in needed if c in df_sym.columns]
        
        df_sym = df_sym[use_cols].copy()
        # Drop rows with NaN in feature_cols or target
        df_sym.dropna(subset=[c for c in feature_cols if c in df_sym.columns] + ['target_5d'], inplace=True)
        
        dfs.append(df_sym)
    
    if not dfs:
        raise ValueError("No dataframes loaded. Check file paths or symbols.")
    
    # Concatenate
    df_all = pd.concat(dfs, axis=0)
    # Sort by date, then symbol
    df_all.sort_values(['date','symbol'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    
    return df_all

################################################################################
# 2) Walk-Forward Split (Time-based) on Combined Data
################################################################################

def walk_forward_split(df, train_years=3, test_months=6):
    """
    Generator yields (train_df, test_df). 
    We do a time-based split ignoring the symbol dimension. 
    All symbols that fall within the date range are included in each split.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    start_date = df['date'].min()
    end_date = df['date'].max()

    current_train_start = start_date
    current_train_end = start_date + pd.DateOffset(years=train_years)
    while current_train_end < end_date:
        current_test_end = current_train_end + pd.DateOffset(months=test_months)
        if current_test_end > end_date:
            current_test_end = end_date

        # Train mask
        train_mask = (df['date'] >= current_train_start) & (df['date'] < current_train_end)
        test_mask  = (df['date'] >= current_train_end)   & (df['date'] < current_test_end)

        train_df = df[train_mask].copy()
        test_df  = df[test_mask].copy()

        if len(train_df) < 50 or len(test_df) < 10:
            break

        yield train_df, test_df

        # Move forward
        current_train_end = current_test_end
        if current_train_end >= end_date:
            break

################################################################################
# 3) Naive Backtest with 5-Day Hold, Cross-Sectional
################################################################################

def cross_sectional_5d_hold(test_df, y_pred):
    """
    Example approach:
    - For each date in test_df, we see which symbols are predicted '1'
    - We sum up their 'future_return_5d' for those rows.
    - This is a simplistic measure: in practice, you'd do a daily rebalancing, 
      or a top-N selection each date, etc.
    
    We'll just sum the returns for all (date, symbol) pairs where the model says '1'.
    """
    test_df = test_df.reset_index(drop=True)
    
    total_return = 0.0
    for i, row in test_df.iterrows():
        if y_pred[i] == 1:
            total_return += row['future_return_5d']
    return total_return

################################################################################
# 4) Main
################################################################################

def main():
    symbols = ["DBC","DIA","GLD","IEF","IWM","LQD","QQQ","SLV",
               "SPY","TLT","UNG","USO","UUP","XLE","XLF"]
    
    # Choose your feature columns
    feature_cols = [
        'rsi_7', 'rsi_14','sma_20','sma_200','ema_50',
        'bb_upper','bb_middle','bb_lower','atr_14',
        'vol_ratio_20','corr_20_spy','daily_return','log_return'
    ]
    
    # 1) Load & combine data
    df_all = load_and_combine_data(symbols, feature_cols)
    print(f"[INFO] Combined dataset shape: {df_all.shape} "
          f"({df_all['symbol'].nunique()} symbols, dates from "
          f"{df_all['date'].min().date()} to {df_all['date'].max().date()})")
    
    # 2) Walk-forward
    all_splits = 0
    all_accuracies = []
    all_returns = []
    
    # We'll store results per split to average later
    for train_df, test_df in walk_forward_split(df_all, train_years=3, test_months=6):
        all_splits += 1

        # Prepare training
        X_train = train_df[feature_cols]
        y_train = train_df['target_5d']

        # Prepare testing
        X_test = test_df[feature_cols]
        y_test = test_df['target_5d']

        # Train one XGBoost model
        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        all_accuracies.append(acc)

        # Simple cross-sectional 5d hold
        # sum up future_return_5d for rows predicted == 1
        test_ret = cross_sectional_5d_hold(test_df, y_pred)
        all_returns.append(test_ret)

        print(f"[Split {all_splits}] Train range: "
              f"{train_df['date'].min().date()} -> {train_df['date'].max().date()} "
              f"Test range: {test_df['date'].min().date()} -> {test_df['date'].max().date()}")
        print(f"  Accuracy: {acc:.4f}, 5dHoldReturn: {test_ret:.4%}")

        # Feature importance for the first split
        if all_splits == 1:
            importances = model.feature_importances_
            feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
            print("[INFO] Feature Importances (first split):")
            for f, imp in feat_imp:
                print(f"  {f}: {imp:.4f}")

    # Summaries
    if all_splits > 0:
        avg_acc = np.mean(all_accuracies)
        total_ret = sum(all_returns)
        print("\n=== Cross-Sectional Walk-Forward Summary ===")
        print(f"Splits: {all_splits}")
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Total 5d Hold Return (sum): {total_ret:.4%}")
    else:
        print("[INFO] No valid splits were generated. Check date ranges or dataset size.")

if __name__ == "__main__":
    main()
