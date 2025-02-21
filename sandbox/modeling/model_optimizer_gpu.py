import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import cudf  # GPU-accelerated DataFrame
import cupy as cp  # GPU-accelerated NumPy
from cuml import RandomForestClassifier  # GPU RandomForest
from cuml.model_selection import train_test_split
import pickle
import time
import logging
from datetime import datetime, timedelta

class GPUModelOptimizer:
    def __init__(self, data, feature_columns):
        """
        Initialize optimizer with data and features
        data: pandas DataFrame
        feature_columns: list of feature names
        """
        self.data = data
        self.feature_columns = feature_columns
        self.results = []
        self.scaler = StandardScaler()
        
    def prepare_data(self, prediction_window=5):
        """Prepare data for GPU computation"""
        # Convert to GPU DataFrame
        try:
            df = cudf.DataFrame(self.data)
        except ImportError:
            logging.warning("CUDA DataFrame failed, using CPU DataFrame")
            df = pd.DataFrame(self.data)
            
        # Create target variable (future returns)
        df['target'] = (df['close'].shift(-prediction_window) > df['close']).astype(int)
        
        # Prepare features
        X = df[self.feature_columns].values
        y = df['target'].values
        
        # Remove NaN values
        mask = ~cp.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
        
    def optimize_random_forest(self):
        """Optimize RandomForest with GPU acceleration"""
        logging.info("Starting GPU RandomForest optimization")
        start_time = time.time()
        
        X, y = self.prepare_data()
        
        # Parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10]
        }
        
        best_score = 0
        best_params = None
        total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * \
                           len(param_grid['max_features']) * len(param_grid['min_samples_split'])
        
        logging.info(f"Testing {total_combinations} parameter combinations")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # GPU-accelerated grid search
        for n_est in param_grid['n_estimators']:
            for depth in param_grid['max_depth']:
                for feat in param_grid['max_features']:
                    for min_split in param_grid['min_samples_split']:
                        scores = []
                        
                        # Initialize GPU RandomForest
                        rf = RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            max_features=feat,
                            min_samples_split=min_split,
                            random_state=42
                        )
                        
                        # Cross-validation
                        for train_idx, val_idx in tscv.split(X):
                            X_train, X_val = X[train_idx], X[val_idx]
                            y_train, y_val = y[train_idx], y[val_idx]
                            
                            rf.fit(X_train, y_train)
                            score = rf.score(X_val, y_val)
                            scores.append(score)
                        
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_params = {
                                'n_estimators': n_est,
                                'max_depth': depth,
                                'max_features': feat,
                                'min_samples_split': min_split
                            }
        
        # Train final model with best parameters
        final_model = RandomForestClassifier(**best_params, random_state=42)
        final_model.fit(X, y)
        
        elapsed_time = time.time() - start_time
        logging.info(f"RandomForest optimization completed in {elapsed_time:.2f} seconds")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score: {best_score:.4f}")
        
        return final_model, best_params, best_score
    
    def optimize_xgboost(self):
        """Optimize XGBoost with GPU acceleration"""
        logging.info("Starting GPU XGBoost optimization")
        start_time = time.time()
        
        X, y = self.prepare_data()
        
        # Parameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Add GPU parameters
        gpu_params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        best_score = 0
        best_params = None
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for max_depth in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                for n_est in param_grid['n_estimators']:
                    for subsample in param_grid['subsample']:
                        for colsample in param_grid['colsample_bytree']:
                            scores = []
                            
                            params = {
                                'max_depth': max_depth,
                                'learning_rate': lr,
                                'n_estimators': n_est,
                                'subsample': subsample,
                                'colsample_bytree': colsample,
                                **gpu_params
                            }
                            
                            # Cross-validation
                            for train_idx, val_idx in tscv.split(X):
                                X_train, X_val = X[train_idx], X[val_idx]
                                y_train, y_val = y[train_idx], y[val_idx]
                                
                                model = xgb.XGBClassifier(**params)
                                model.fit(X_train, y_train)
                                score = model.score(X_val, y_val)
                                scores.append(score)
                            
                            avg_score = np.mean(scores)
                            if avg_score > best_score:
                                best_score = avg_score
                                best_params = params
        
        # Train final model with best parameters
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X, y)
        
        elapsed_time = time.time() - start_time
        logging.info(f"XGBoost optimization completed in {elapsed_time:.2f} seconds")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score: {best_score:.4f}")
        
        return final_model, best_params, best_score

def run_gpu_optimization(data, symbol):
    """Main function to run GPU-accelerated optimization"""
    
    # Define features
    feature_columns = [
        'returns', 'returns_volatility', 
        'RSI_14', 'MACD', 'MACD_Signal',
        'SMA_10', 'EMA_5', 'ATR_14',
        'Bollinger_High_20', 'Bollinger_Low_20'
    ]
    
    # Initialize optimizer
    optimizer = GPUModelOptimizer(data, feature_columns)
    
    # Run optimizations
    rf_model, rf_params, rf_score = optimizer.optimize_random_forest()
    xgb_model, xgb_params, xgb_score = optimizer.optimize_xgboost()
    
    # Save results
    results = {
        'symbol': symbol,
        'optimization_date': datetime.now(),
        'rf_model': rf_model,
        'rf_params': rf_params,
        'rf_score': rf_score,
        'xgb_model': xgb_model,
        'xgb_params': xgb_params,
        'xgb_score': xgb_score,
        'feature_columns': feature_columns,
        'scaler': optimizer.scaler
    }
    
    # Save to file
    filename = f"gpu_optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    return results