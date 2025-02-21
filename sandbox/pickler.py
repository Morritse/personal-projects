import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import yfinance as yf
from datetime import datetime, timedelta
import logging

# The optimized parameters from Colab
OPTIMIZED_PARAMS = {
    'rf_params': {
        'n_estimators': 200,
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 4
    },
    'xgb_params': {
        'max_depth': 7,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 1.0,
        'colsample_bytree': 0.9,
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False  # Add this line
    },
    'feature_columns': [
        'returns', 'returns_volatility', 
        'RSI_14', 'MACD', 'MACD_Signal',
        'SMA_10', 'EMA_5', 'ATR_14',
        'BB_middle', 'BB_upper', 'BB_lower'
    ]
}

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

def create_local_models(symbol='SPY', lookback_days=500):
    """Create and save models using optimized parameters"""
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    print(f"Fetching data for {symbol}")
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    
    # Calculate indicators
    df = calculate_technical_indicators(df)
    df = df.dropna()
    
    # Prepare features
    X = df[OPTIMIZED_PARAMS['feature_columns']].values
    y = (df['Close'].shift(-5) > df['Close']).astype(int)  # 5-day prediction window
    
    # Remove NaN values
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create and train models with optimized parameters
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(**OPTIMIZED_PARAMS['rf_params'], random_state=42)
    rf_model.fit(X, y)
    
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(**OPTIMIZED_PARAMS['xgb_params'])
    xgb_model.fit(X, y)
    
    # Create results dictionary
    results = {
        'symbol': symbol,
        'optimization_date': datetime.now(),
        'rf_model': rf_model,
        'rf_params': OPTIMIZED_PARAMS['rf_params'],
        'xgb_model': xgb_model,
        'xgb_params': OPTIMIZED_PARAMS['xgb_params'],
        'feature_columns': OPTIMIZED_PARAMS['feature_columns'],
        'scaler': scaler
    }
    
    # Save with protocol 4 for Python 3.6 compatibility
    filename = f"local_models_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
    print(f"Saving models to {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(results, f, protocol=4)
    
    return results

if __name__ == "__main__":
    create_local_models('SPY')