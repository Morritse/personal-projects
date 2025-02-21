import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib  # Using joblib instead of pickle
import time
import logging
from datetime import datetime, timedelta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Updated for standard tqdm

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
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
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
        
        # Drop rows with NaN values resulting from calculations
        df.dropna(inplace=True)
        
        return df
        
    except Exception as e:
        logging.error(f"Error calculating technical indicators: {str(e)}")
        raise

class ModelOptimizer:
    def __init__(self, data, feature_columns):
        self.data = data
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        
    def prepare_data(self, prediction_window=5):
        """Prepare data for model training"""
        df = pd.DataFrame(self.data)
        
        # Create target variable (future returns)
        df['target'] = (df['Close'].shift(-prediction_window) > df['Close']).astype(int)
        
        # Prepare features
        X = df[self.feature_columns].values
        y = df['target'].values
        
        # Remove NaN values
        valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        self.X = X
        self.y = y
        
        return X, y
    
    def optimize_random_forest(self):
        """Optimize RandomForest parameters using RandomizedSearchCV"""
        logging.info("Starting RandomForest optimization")
        start_time = time.time()
        
        X, y = self.X, self.y
        
        param_dist = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        rf = RandomForestClassifier(random_state=42)
        
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=50,
            cv=tscv,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        best_rf = random_search.best_estimator_
        best_score = random_search.best_score_
        best_params = random_search.best_params_
        
        elapsed_time = time.time() - start_time
        logging.info(f"RandomForest optimization completed in {elapsed_time:.2f} seconds")
        logging.info(f"Best RandomForest score: {best_score:.4f}")
        logging.info(f"Best RandomForest parameters: {best_params}")
        
        return best_rf, best_params, best_score
    
    def optimize_xgboost(self):
        """Optimize XGBoost parameters using RandomizedSearchCV with GPU acceleration"""
        logging.info("Starting XGBoost optimization")
        start_time = time.time()
        
        X, y = self.X, self.y
        
        param_dist = {
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.3],
            'n_estimators': [100, 200, 300, 400],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [1, 1.5, 2]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        xgb_clf = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,  # This parameter is deprecated
            tree_method='hist',
            device='cuda',  # Use GPU
            random_state=42
        )
        
        random_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_dist,
            n_iter=50,
            cv=tscv,
            scoring='accuracy',
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        best_xgb = random_search.best_estimator_
        best_score = random_search.best_score_
        best_params = random_search.best_params_
        
        elapsed_time = time.time() - start_time
        logging.info(f"XGBoost optimization completed in {elapsed_time:.2f} seconds")
        logging.info(f"Best XGBoost score: {best_score:.4f}")
        logging.info(f"Best XGBoost parameters: {best_params}")
        
        return best_xgb, best_params, best_score

def fetch_data(symbol, lookback_days=500):
    """Fetch and prepare data for a symbol"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    logging.info(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}")
    
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
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
    optimizer.prepare_data()
    
    # Run optimizations
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
    
    # Save results using joblib
    filename = f"optimization_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.joblib"
    joblib.dump(results, filename)
    logging.info(f"Optimization results saved to {filename}")
    
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
        logging.error(f"Error processing {symbol}: {str(e)}")
        continue

# Plot results comparison
plt.figure(figsize=(8, 6))
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


from google.colab import drive
drive.mount('/content/drive')
for symbol, result in results.items():
    filename = f"/content/drive/MyDrive/trading_optimization_{symbol}.joblib"
    joblib.dump(result, filename)
    logging.info(f"Saved {symbol} optimization to Google Drive.")
