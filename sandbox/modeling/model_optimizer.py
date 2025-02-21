import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

class ModelOptimizer:
    def __init__(self, data, feature_columns):
        self.data = data
        self.feature_columns = feature_columns
        self.results = []
        
    def optimize_random_forest(self):
        """Optimize RandomForest parameters"""
        # Define param_grid at the start of the method
        param_grid = {
            #'n_estimators': [100, 200, 300],
            #'max_depth': [3, 5, 7, 10],
            #'min_samples_split': [2, 5, 10],
            #'min_samples_leaf': [1, 2, 4],
            'n_estimators': [100, 200],      # Reduced from [100, 200, 300]
            'max_depth': [3, 7],             # Reduced from [3, 5, 7, 10]
            'min_samples_split': [2, 10],    # Reduced from [2, 5, 10]
            'min_samples_leaf': [1, 4]       # Reduced from [1, 2, 4]
        }
        
        print("Starting Random Forest grid search...")
        print(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])} combinations")
        
        rf = RandomForestClassifier(random_state=42)
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose = 3
        )
        
        X = self.data[self.feature_columns]
        y = (self.data['close'].shift(-1) > self.data['close']).astype(int)
        
        # Remove NaN values
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        grid_search.fit(X, y)
        
        self.results.append({
            'model': 'RandomForest',
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        })
        
        return grid_search.best_estimator_
        
    def optimize_xgboost(self):
        """Optimize XGBoost parameters"""
        param_grid = {
           # 'n_estimators': [100, 200, 300],
            #'max_depth': [3, 5, 7],
            #'learning_rate': [0.01, 0.1, 0.3],
            #'subsample': [0.8, 0.9, 1.0],
            #'colsample_bytree': [0.8, 0.9, 1.0]
            'n_estimators': [100, 300],      # Reduced from [100, 200, 300]
            'max_depth': [3, 7],             # Reduced from [3, 5, 7]
            'learning_rate': [0.01, 0.3],    # Reduced from [0.01, 0.1, 0.3]
            'subsample': [0.8, 1.0],         # Reduced from [0.8, 0.9, 1.0]
            'colsample_bytree': [0.8, 1.0]  
        }
        
        print("Starting XGBoost grid search...")
        print(f"Testing {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} combinations")
        
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose = 3
        )
        
        X = self.data[self.feature_columns]
        y = (self.data['close'].shift(-1) > self.data['close']).astype(int)
        
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        grid_search.fit(X, y)
        
        self.results.append({
            'model': 'XGBoost',
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        })
        
        return grid_search.best_estimator_

    def optimize_trading_params(self, model, param_ranges):
        """Optimize trading parameters (e.g., lookback periods, thresholds)"""
        results = []
        
        # Generate all combinations of parameters
        param_combinations = list(product(*param_ranges.values()))
        
        for params in param_combinations:
            param_dict = dict(zip(param_ranges.keys(), params))
            
            # Apply parameters to generate features
            temp_data = self.data.copy()
            
            # Example: Adjust moving averages based on parameters
            if 'sma_period' in param_dict:
                temp_data[f'sma_{param_dict["sma_period"]}'] = temp_data['close'].rolling(
                    window=param_dict['sma_period']
                ).mean()
            
            if 'rsi_period' in param_dict:
                # Calculate RSI with different periods
                delta = temp_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(
                    window=param_dict['rsi_period']
                ).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(
                    window=param_dict['rsi_period']
                ).mean()
                rs = gain / loss
                temp_data['rsi'] = 100 - (100 / (1 + rs))
            
            # Run backtest with these parameters
            backtest_results = self.run_backtest(model, temp_data, param_dict)
            
            results.append({
                'params': param_dict,
                'sharpe_ratio': backtest_results['sharpe_ratio'],
                'total_return': backtest_results['total_return'],
                'max_drawdown': backtest_results['max_drawdown']
            })
        
        self.trading_results = pd.DataFrame(results)
        return self.trading_results
    
    def plot_optimization_results(self):
        """Plot optimization results"""
        # Plot model accuracies
        plt.figure(figsize=(10, 6))
        scores = [result['best_score'] for result in self.results]
        models = [result['model'] for result in self.results]
        
        plt.bar(models, scores)
        plt.title('Model Accuracies')
        plt.ylabel('Accuracy Score')
        plt.ylim(0, 1)
        
        plt.show()
        
        # Plot trading parameter results if available
        if hasattr(self, 'trading_results'):
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            sns.boxplot(data=self.trading_results, y='sharpe_ratio')
            plt.title('Sharpe Ratio Distribution')
            
            plt.subplot(1, 3, 2)
            sns.boxplot(data=self.trading_results, y='total_return')
            plt.title('Total Return Distribution')
            
            plt.subplot(1, 3, 3)
            sns.boxplot(data=self.trading_results, y='max_drawdown')
            plt.title('Max Drawdown Distribution')
            
            plt.tight_layout()
            plt.show()
    
    def run_backtest(self, model, data, params):
        """Run backtest with given parameters"""
        # Initialize portfolio
        portfolio_value = 100000
        position = 0
        returns = []
        
        # Trading loop
        for i in range(len(data)-1):
            if i < max(params.values()):  # Skip until we have enough data
                continue
                
            # Get model prediction
            features = data[self.feature_columns].iloc[i:i+1]
            pred = model.predict(features)[0]
            
            # Simple trading logic
            if pred == 1 and position == 0:
                position = 1
            elif pred == 0 and position == 1:
                position = 0
            
            # Calculate returns
            if position == 1:
                daily_return = data['close'].iloc[i+1] / data['close'].iloc[i] - 1
                returns.append(daily_return)
        
        # Calculate metrics
        returns = pd.Series(returns)
        metrics = {
            'sharpe_ratio': np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 0 else 0,
            'total_return': (1 + returns).prod() - 1 if len(returns) > 0 else 0,
            'max_drawdown': (returns.cumsum() - returns.cumsum().expanding().max()).min()
        }
        
        return metrics