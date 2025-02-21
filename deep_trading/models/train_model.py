import os
import numpy as np
import pandas as pd
import joblib
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from momentum_ai_trading.utils.config import (
    PROCESSED_DATA_PATH, 
    MODEL_PATH, 
    RESULTS_PATH
)
from momentum_ai_trading.utils.feature_engineering import prepare_features

def load_data(symbol='AAPL'):
    """Load and preprocess data for model training."""
    data_path = os.path.join(PROCESSED_DATA_PATH, f"{symbol}_processed_daily.csv")
    df = pd.read_csv(data_path)
    
    # Prepare features and target
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for future use
    joblib.dump(scaler, f"{MODEL_PATH}_scaler.pkl")
    joblib.dump(X.columns.tolist(), f"{MODEL_PATH}_features.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def objective(trial):
    """Objective function for Optuna hyperparameter optimization."""
    # Hyperparameters to optimize
    params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    
    # Load data
    X_train, X_test, y_train, y_test, _ = load_data()
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Train model
    model = lgb.train(params, train_data, num_boost_round=100)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return mse

def optimize_hyperparameters(n_trials=100):
    """Run Optuna hyperparameter optimization."""
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (MSE): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return trial.params

def train_final_model(best_params=None):
    """Train final model with best hyperparameters."""
    if best_params is None:
        best_params = optimize_hyperparameters()
    
    # Load data
    X_train, X_test, y_train, y_test, scaler = load_data()
    
    # Prepare LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Final model training with best params
    final_params = {
        'objective': 'regression',
        'metric': 'mse',
        'boosting_type': 'gbdt',
        **best_params
    }
    
    model = lgb.train(final_params, train_data, num_boost_round=200)
    
    # Save model
    model.save_model(f"{MODEL_PATH}.txt")
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    
    # Performance metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Save performance metrics
    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }
    
    with open(f"{MODEL_PATH}_metadata.pkl", 'wb') as f:
        joblib.dump(metrics, f)
    
    # Feature importance
    feature_importance = model.feature_importance()
    feature_names = joblib.load(f"{MODEL_PATH}_features.pkl")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv(f"{MODEL_PATH}_importance.csv", index=False)
    
    return model, metrics

def main():
    """Main function to run model training and optimization."""
    best_params = optimize_hyperparameters()
    model, metrics = train_final_model(best_params)
    
    print("\nModel Training Complete:")
    print("Best Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

if __name__ == "__main__":
    main()
