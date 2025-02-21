# Check GPU is enabled and install packages
!nvidia-smi


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle
import time
import logging
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataset"""
    try:
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Moving Averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        
        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        df['BB_middle'] = rolling_mean
        df['BB_upper'] = rolling_mean + (rolling_std * 2)
        df['BB_lower'] = rolling_mean - (rolling_std * 2)
        
        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR_14'] = true_range.rolling(14).mean()
        
        # Additional features
        df['returns'] = df['Close'].pct_change()
        df['returns_volatility'] = df['returns'].rolling(window=20).std()
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {str(e)}")
        raise

class ModelOptimizer:
    def __init__(self, data, feature_columns):
        self.data = data
        self.feature_columns = feature_columns
        self.results = []
        self.scaler = StandardScaler()
        
    def prepare_data(self, prediction_window=5):
        """Prepare data for model training"""
        df = pd.DataFrame(self.data)
        
        # Create target variable (future returns)
        df['target'] = (df['Close'].shift(-prediction_window) > df['Close']).astype(int)
        
        # Prepare features
        X = df[self.feature_columns].values
        y = df['target'].values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def optimize_random_forest(self):
        """Optimize RandomForest parameters"""
        logging.info("Starting RandomForest optimization")
        start_time = time.time()
        
        X, y = self.prepare_data()
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        best_score = 0
        best_params = None
        
        total_combinations = (len(param_grid['n_estimators']) * 
                            len(param_grid['max_depth']) * 
                            len(param_grid['min_samples_split']) * 
                            len(param_grid['min_samples_leaf']))
        
        print(f"Testing {total_combinations} RandomForest parameter combinations")
        pbar = tqdm(total=total_combinations)
        
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            
            for n_est in param_grid['n_estimators']:
                for depth in param_grid['max_depth']:
                    for min_split in param_grid['min_samples_split']:
                        for min_leaf in param_grid['min_samples_leaf']:
                            scores = []
                            
                            rf = RandomForestClassifier(
                                n_estimators=n_est,
                                max_depth=depth,
                                min_samples_split=min_split,
                                min_samples_leaf=min_leaf,
                                random_state=42
                            )
                            
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
                                    'min_samples_split': min_split,
                                    'min_samples_leaf': min_leaf
                                }
                                print(f"\nNew best RF score: {best_score:.4f}")
                                print(f"Parameters: {best_params}")
                            
                            pbar.update(1)
        
        except Exception as e:
            print(f"Error during RF optimization: {str(e)}")
            raise e
            
        finally:
            pbar.close()
        
        final_model = RandomForestClassifier(**best_params, random_state=42)
        final_model.fit(X, y)
        
        elapsed_time = time.time() - start_time
        logging.info(f"RandomForest optimization completed in {elapsed_time:.2f} seconds")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score: {best_score:.4f}")
        
        return final_model, best_params, best_score
    
    def optimize_xgboost(self):
        """Optimize XGBoost with GPU acceleration"""
        logging.info("Starting XGBoost optimization")
        start_time = time.time()
        
        X, y = self.prepare_data()
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Updated GPU parameters
        gpu_params = {
            'device': 'cuda',
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        best_score = 0
        best_params = None
        
        total_combinations = (len(param_grid['max_depth']) * 
                            len(param_grid['learning_rate']) * 
                            len(param_grid['n_estimators']) * 
                            len(param_grid['subsample']) * 
                            len(param_grid['colsample_bytree']))
        
        print(f"Testing {total_combinations} XGBoost parameter combinations")
        pbar = tqdm(total=total_combinations)
        
        try:
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
                                    print(f"\nNew best XGB score: {best_score:.4f}")
                                    print(f"Parameters: {best_params}")
                                
                                pbar.update(1)
        
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            raise e
            
        finally:
            pbar.close()
        
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(X, y)
        
        elapsed_time = time.time() - start_time
        logging.info(f"XGBoost optimization completed in {elapsed_time:.2f} seconds")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score: {best_score:.4f}")
        
        return final_model, best_params, best_score

def fetch_data(symbol, lookback_days=500):
    """Fetch and prepare data for a symbol"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
    
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Remove NaN values
    df = df.dropna()
    
    logging.info(f"Prepared {len(df)} rows of data for {symbol}")
    return df

def run_optimization(data, symbol):
    """Main function to run optimization"""
    
    feature_columns = [
        'returns', 'returns_volatility', 
        'RSI_14', 'MACD', 'MACD_Signal',
        'SMA_10', 'EMA_5', 'ATR_14',
        'BB_middle', 'BB_upper', 'BB_lower'
    ]
    
    optimizer = ModelOptimizer(data, feature_columns)
    
    # Run both optimizations
    rf_model, rf_params, rf_score = optimizer.optimize_random_forest()
    xgb_model, xgb_params, xgb_score = optimizer.optimize_xgboost()
    
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
    
    # Save results
    filename = f"optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    
    return results

# Run optimization for symbols
symbols = ['SPY']  # Add or modify symbols as needed
results = {}

for symbol in symbols:
    print(f"\n{'='*50}")
    print(f"Optimizing {symbol}")
    print('='*50)
    
    try:
        data = fetch_data(symbol)
        results[symbol] = run_optimization(data, symbol)
        
        print(f"\nResults for {symbol}:")
        print(f"RandomForest Score: {results[symbol]['rf_score']:.4f}")
        print(f"XGBoost Score: {results[symbol]['xgb_score']:.4f}")
        print(f"Best RF Parameters: {results[symbol]['rf_params']}")
        print(f"Best XGB Parameters: {results[symbol]['xgb_params']}")
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        continue

# Plot results comparison
plt.figure(figsize=(12, 6))
symbols_list = list(results.keys())
rf_scores = [results[s]['rf_score'] for s in symbols_list]
xgb_scores = [results[s]['xgb_score'] for s in symbols_list]

x = np.arange(len(symbols_list))
width = 0.35

plt.bar(x - width/2, rf_scores, width, label='RandomForest')
plt.bar(x + width/2, xgb_scores, width, label='XGBoost')

plt.xlabel('Symbols')
plt.ylabel('Model Score')
plt.title('Model Performance Comparison by Symbol')
plt.xticks(x, symbols_list)
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')
for symbol, result in results.items():
    filename = f"/content/drive/MyDrive/trading_optimization_{symbol}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(result, f)