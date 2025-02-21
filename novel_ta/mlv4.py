"""
mlv4.py - Updated version demonstrating:
 - cross-sectional single model
 - walk-forward splits
 - daily rebalancing simulation with position tracking
 - 5-day classification label (target_5d)
"""

import pandas as pd
import numpy as np
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

###############################################################################
# 1) Master Column Set
###############################################################################

MASTER_COLS = [
    'date', 'symbol', 'close',
    # Target columns
    'target_5d', 'future_return_5d',
    # Features grouped by type
    'rsi_7', 'rsi_14',
    'sma_20', 'sma_200', 'ema_50',
    'bb_upper', 'bb_middle', 'bb_lower',
    'atr_14', 'vol_ratio_20', 'corr_20_spy'
]

###############################################################################
# 1a) Load & Combine Data
###############################################################################

def load_and_combine_data(symbols, base_feature_cols):
    """
    For each symbol:
      1) Load CSV 
      2) Reindex columns to MASTER_COLS
      3) Drop rows missing critical data
      4) Add derived features
      5) Concatenate into one big DataFrame
    """
    dfs = []
    for sym in symbols:
        filepath = f"featuresv2_data/{sym}_with_features_v2_1d.csv"
        if not os.path.exists(filepath):
            print(f"[WARN] {filepath} not found. Skipping {sym}.")
            continue
        
        # 1) Read CSV
        df_sym = pd.read_csv(filepath, parse_dates=['date'])
        df_sym['symbol'] = sym
        
        # 2) Reindex to ensure consistent columns
        all_needed = list(set(MASTER_COLS + base_feature_cols))
        df_sym = df_sym.reindex(columns=all_needed)
        
        # 3) Drop rows missing crucial columns
        must_have = ['target_5d', 'future_return_5d', 'close', 'bb_middle'] + [c for c in base_feature_cols if c != 'corr_20_spy']
        df_sym.dropna(subset=must_have, inplace=True)
        
        # 4) Add derived features
        df_sym['sma_ratio_20_200'] = df_sym['sma_20'] / df_sym['sma_200'] - 1
        df_sym['bb_position'] = (df_sym['close'] - df_sym['bb_middle']) / (df_sym['bb_upper'] - df_sym['bb_lower'])
        
        df_sym.reset_index(drop=True, inplace=True)
        dfs.append(df_sym)
    
    if not dfs:
        raise ValueError("No dataframes loaded. Check file paths or symbols.")
    
    # 5) Concatenate them
    df_all = pd.concat(dfs, axis=0, join='outer', ignore_index=True)
    df_all.sort_values(['date','symbol'], inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    
    return df_all

###############################################################################
# 2) Walk-Forward Split
###############################################################################

def walk_forward_split(df, train_years=3, test_months=6):
    """
    Yields (train_df, test_df) in a rolling time-based manner.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)

    start_date = df['date'].min()
    end_date   = df['date'].max()

    current_train_start = start_date
    current_train_end   = start_date + pd.DateOffset(years=train_years)

    while current_train_end < end_date:
        current_test_end = current_train_end + pd.DateOffset(months=test_months)
        if current_test_end > end_date:
            current_test_end = end_date

        train_mask = (df['date'] >= current_train_start) & (df['date'] < current_train_end)
        test_mask  = (df['date'] >= current_train_end)   & (df['date'] < current_test_end)

        train_df = df[train_mask].copy()
        test_df  = df[test_mask].copy()

        if len(train_df) < 50 or len(test_df) < 10:
            break

        yield train_df, test_df

        current_train_end = current_test_end
        if current_train_end >= end_date:
            break

###############################################################################
# 3) Portfolio Simulation with Position Tracking
###############################################################################

def portfolio_simulation(test_df, y_pred_proba, prob_threshold=0.5, top_k=5, cost_per_trade=0.0005):
    """
    Enhanced portfolio simulation:
    - Uses probability threshold for filtering
    - Tracks positions to avoid unnecessary trading
    - Only applies transaction costs on position changes
    - Uses future_return_5d for returns (aligned with target)
    """
    test_df = test_df.copy().reset_index(drop=True)
    test_df['pred_proba'] = y_pred_proba

    unique_dates = sorted(test_df['date'].unique())
    portfolio_value = 1.0
    daily_values = []
    current_positions = set()  # Track currently held symbols

    for d in unique_dates:
        day_rows = test_df[test_df['date'] == d].copy()
        if len(day_rows) == 0:
            daily_values.append((d, portfolio_value))
            continue
        
        # Filter by probability threshold first
        day_rows = day_rows[day_rows['pred_proba'] >= prob_threshold]
        
        # Get top predictions for today
        day_rows = day_rows.sort_values('pred_proba', ascending=False)
        new_picks = set(day_rows.head(top_k)['symbol'])
        
        if len(new_picks) == 0:
            daily_values.append((d, portfolio_value))
            continue

        # Calculate position changes
        positions_to_add = new_picks - current_positions
        positions_to_remove = current_positions - new_picks
        
        # Apply transaction costs only on changes
        total_trades = len(positions_to_add) + len(positions_to_remove)
        trade_cost = portfolio_value * (cost_per_trade * total_trades)
        
        # Calculate returns using future_return_5d for selected positions
        selected_rows = day_rows[day_rows['symbol'].isin(new_picks)]
        avg_return = selected_rows['future_return_5d'].mean()
        
        # Update portfolio value
        if not np.isnan(avg_return):
            portfolio_value = portfolio_value * (1 + avg_return) - trade_cost
        
        daily_values.append((d, portfolio_value))
        current_positions = new_picks

    return daily_values

###############################################################################
# 4) Main
###############################################################################

def main():
    # Symbol set
    symbols = ["DBC", "DIA", "GLD", "IEF", "IWM", 
               "LQD", "QQQ", "SLV", "SPY", "TLT", 
               "UNG", "USO", "UUP", "XLE", "XLF"]

    # Base feature columns grouped by type
    base_feature_cols = [
        # Price Momentum (shorter-term)
        'rsi_7', 'rsi_14',
        
        # Trend Following (longer-term)
        'sma_20', 'sma_200', 'ema_50',
        
        # Mean Reversion
        'bb_upper', 'bb_lower',
        
        # Volatility
        'atr_14',
        
        # Market Relationship
        'vol_ratio_20', 'corr_20_spy'
    ]
    
    # Load & combine data
    df_all = load_and_combine_data(symbols, base_feature_cols)
    
    # Complete feature set including derived features
    feature_cols = base_feature_cols + ['sma_ratio_20_200', 'bb_position']
    
    print(f"[INFO] Combined dataset shape: {df_all.shape} "
          f"({df_all['symbol'].nunique()} symbols, dates from "
          f"{df_all['date'].min().date()} to {df_all['date'].max().date()})")

    # Walk-forward testing
    all_splits = 0
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    final_portvals = []

    for train_df, test_df in walk_forward_split(df_all, train_years=3, test_months=6):
        all_splits += 1

        # Prepare data with scaling
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(train_df[feature_cols].fillna(0)), 
                             columns=feature_cols)
        y_train = train_df['target_5d']
        
        X_test = pd.DataFrame(scaler.transform(test_df[feature_cols].fillna(0)), 
                            columns=feature_cols)
        y_test = test_df['target_5d']

        # Grid search setup with regularization
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [2, 3, 4],
            'learning_rate': [0.01, 0.03],
            'min_child_weight': [5, 7],
            'subsample': [0.8],
            'colsample_bytree': [0.8],
            'gamma': [1],  # min split loss
            'reg_alpha': [1],  # L1 regularization
            'reg_lambda': [1]   # L2 regularization
        }
        
        # Multiple validation splits for more robust evaluation
        val_splits = []
        split_points = [0.6, 0.7, 0.8]  # Try different split ratios
        
        for split_ratio in split_points:
            train_size = int(len(X_train) * split_ratio)
            X_train_inner = X_train.iloc[:train_size]
            y_train_inner = y_train.iloc[:train_size]
            X_val = X_train.iloc[train_size:]
            y_val = y_train.iloc[train_size:]
            val_splits.append((X_train_inner, y_train_inner, X_val, y_val))
        
        # Grid search
        print("\n[INFO] Performing grid search...")
        best_score = 0
        best_params = None
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        current_combo = 0
        
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for min_child in param_grid['min_child_weight']:
                        current_combo += 1
                        if all_splits == 1:
                            print(f"\nTrying combination {current_combo}/{total_combinations}:")
                            print(f"  n_estimators={n_est}, max_depth={depth}, "
                                  f"learning_rate={lr}, min_child_weight={min_child}")
                        
                        model = XGBClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            min_child_weight=min_child,
                            subsample=param_grid['subsample'][0],
                            colsample_bytree=param_grid['colsample_bytree'][0],
                            gamma=param_grid['gamma'][0],
                            reg_alpha=param_grid['reg_alpha'][0],
                            reg_lambda=param_grid['reg_lambda'][0],
                            random_state=42
                        )
                        
                        # Average score across multiple validation splits
                        split_scores = []
                        for X_tr, y_tr, X_v, y_v in val_splits:
                            model.fit(X_tr, y_tr)
                            y_pred = model.predict(X_v)
                            split_scores.append(accuracy_score(y_v, y_pred))
                        score = np.mean(split_scores)
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'learning_rate': lr,
                                'min_child_weight': min_child,
                                'subsample': param_grid['subsample'][0],
                                'colsample_bytree': param_grid['colsample_bytree'][0],
                                'gamma': param_grid['gamma'][0],
                                'reg_alpha': param_grid['reg_alpha'][0],
                                'reg_lambda': param_grid['reg_lambda'][0]
                            }
        
        # Train final model
        print("\n[INFO] Best parameters found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"  Validation accuracy: {best_score:.4f}")
        
        model = XGBClassifier(**best_params, random_state=42)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred_proba = model.predict_proba(X_test)[:,1]
        
        # Find optimal probability threshold on validation data
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_threshold = 0.5
        best_val_score = 0
        
        for threshold in thresholds:
            val_scores = []
            for _, _, X_v, y_v in val_splits:
                y_v_pred_proba = model.predict_proba(X_v)[:,1]
                y_v_pred = (y_v_pred_proba >= threshold).astype(int)
                val_scores.append(accuracy_score(y_v, y_v_pred))
            avg_score = np.mean(val_scores)
            if avg_score > best_val_score:
                best_val_score = avg_score
                best_threshold = threshold
        
        print(f"  Optimal probability threshold: {best_threshold:.2f}")
        
        # Final predictions using optimal threshold
        y_pred_class = (y_pred_proba >= best_threshold).astype(int)
        acc = accuracy_score(y_test, y_pred_class)
        prec = precision_score(y_test, y_pred_class)
        rec = recall_score(y_test, y_pred_class)
        
        all_accuracies.append(acc)
        all_precisions.append(prec)
        all_recalls.append(rec)

        # Portfolio simulation with optimal threshold
        daily_vals = portfolio_simulation(test_df, y_pred_proba, 
                                       prob_threshold=best_threshold,
                                       top_k=5, cost_per_trade=0.0005)
        if daily_vals:
            final_portfolio = daily_vals[-1][1]
            final_portvals.append(final_portfolio)
        else:
            final_portvals.append(1.0)

        # Calculate annualized return
        test_days = (test_df['date'].max() - test_df['date'].min()).days
        annualized_return = (final_portvals[-1] ** (365/test_days)) - 1 if test_days > 0 else 0
        
        print(f"[Split {all_splits}] Train: {train_df['date'].min().date()} -> {train_df['date'].max().date()}, "
              f"Test: {test_df['date'].min().date()} -> {test_df['date'].max().date()}")
        print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
        print(f"  Test Period Return: {(final_portvals[-1] - 1)*100:.2f}%")
        print(f"  Annualized Return: {annualized_return*100:.2f}% YoY")

        if all_splits == 1:
            importances = model.feature_importances_
            feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
            print("[INFO] Feature Importances (first split):")
            for f, imp in feat_imp:
                print(f"  {f}: {imp:.4f}")

    # Summary statistics
    if all_splits > 0:
        avg_acc = np.mean(all_accuracies)
        avg_prec = np.mean(all_precisions)
        avg_rec = np.mean(all_recalls)
        avg_port = np.mean(final_portvals)
        print("\n=== Cross-Sectional Walk-Forward Summary ===")
        print(f"Splits: {all_splits}")
        print(f"Average Accuracy: {avg_acc:.4f}")
        print(f"Average Precision: {avg_prec:.4f}")
        print(f"Average Recall: {avg_rec:.4f}")
        print(f"Average Portfolio Return: {(avg_port - 1)*100:.2f}%")
        print(f"Average Annualized Return: {np.mean([((pv ** (365/((test_df['date'].max() - test_df['date'].min()).days))) - 1) for pv in final_portvals])*100:.2f}% YoY")
        print("NOTE: Returns are gross of transaction costs")
    else:
        print("[INFO] No valid splits generated.")

if __name__ == "__main__":
    main()
