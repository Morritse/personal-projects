import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from strategy.vwap_obv_strategy import VWAPOBVCrossover
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, RobustScaler
import matplotlib.pyplot as plt

def fetch_historical_data(days=365):
    """Fetch and prepare historical data with indicators."""
    # Load environment variables
    load_dotenv()
    
    # Setup API connection
    api_key = os.getenv('ALPACA_API_KEY')
    api_secret = os.getenv('ALPACA_SECRET_KEY')
    base_url = 'https://paper-api.alpaca.markets'
    
    if not api_key or not api_secret:
        raise ValueError("API credentials not found in environment variables")
        
    api = tradeapi.REST(
        api_key,
        api_secret,
        base_url=base_url,
        api_version='v2'
    )
    
    # Get historical data
    end = datetime.now()
    start = end - timedelta(days=days)
    
    print(f"\nFetching {days} days of historical data for JNJ...")
    bars = api.get_bars(
        'JNJ',
        tradeapi.TimeFrame.Hour,
        start.strftime('%Y-%m-%d'),
        end.strftime('%Y-%m-%d'),
        adjustment='raw'
    ).df
    
    print(f"Got {len(bars)} hours of data")
    
    # Initialize strategy for indicators
    strategy = VWAPOBVCrossover(instruments=['JNJ'])
    
    # Calculate indicators
    print("Calculating indicators...")
    df = bars.copy()
    
    # Base indicators
    df['vwap'] = strategy.get_vwap(df)
    df['obv'] = strategy.get_obv(df)
    df['mfi'] = strategy.get_mfi(df)
    df['ema'] = strategy.get_ema(df)
    df['atr'] = strategy.get_atr(df)
    df['vol_score'] = strategy.get_volume_pattern_score(df)
    
    # Add temporal features based on our analysis
    print("Adding temporal features...")
    
    # 1. Volume ratios at key lags (1-3 hours where we found asymmetry)
    for lag in [1, 2, 3]:
        df[f'vol_ratio_{lag}'] = df['volume'] / df['volume'].shift(lag)
        # Z-score of volume ratio
        ratio = df[f'vol_ratio_{lag}']
        df[f'vol_ratio_{lag}_zscore'] = (ratio - ratio.rolling(100).mean()) / ratio.rolling(100).std()
    
    # 2. Price momentum at key lags
    for lag in [1, 3, 40]:  # Including the lag-40 autocorrelation
        df[f'price_mom_{lag}'] = df['close'].pct_change(lag)
        
    # 3. Volume-weighted price changes
    df['vw_price_change'] = df['close'].pct_change() * (df['volume'] / df['volume'].rolling(3).mean())
    
    # 4. Rolling volatility at different windows
    for window in [3, 40]:  # 3 hours (short-term) and 40 hours (autocorrelation lag)
        df[f'volatility_{window}h'] = df['close'].pct_change().rolling(window).std()
    
    # 5. Volume patterns (simpler approach)
    df['volume_ma3'] = df['volume'].rolling(3).mean()
    df['volume_ma40'] = df['volume'].rolling(40).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma3']
    df['volume_trend'] = df['volume_ma3'] / df['volume_ma40']
    
    # Calculate target: Next bar's close price
    df['next_close'] = df['close'].shift(-1)
    
    # Print distribution info
    print("\nPrice distribution:")
    print(df['close'].describe())
    print("\nNext close distribution:")
    print(df['next_close'].describe())
    
    # Print data quality info
    print("\nMissing values before cleanup:")
    print(df.isnull().sum())
    
    # Handle NaN values more carefully
    # First, forward fill indicators that should maintain last value
    indicators_ffill = ['ema', 'obv', 'vwap']
    df[indicators_ffill] = df[indicators_ffill].fillna(method='ffill')
    
    # For rolling calculations, we need a minimum window
    min_window = 100  # Maximum of our rolling windows
    df = df.iloc[min_window:]
    
    # Now drop any remaining NaN values
    df = df.dropna()
    
    print("\nMissing values after cleanup:")
    print(df.isnull().sum())
    print(f"\nFinal row count: {len(df)}")
    
    return df

def create_supervised_data(df, feature_cols, target_col='next_close', lookback=60):
    """Create sequences for supervised learning."""
    X, y = [], []
    
    print("\nCreating sequences...")
    print(f"Feature columns: {feature_cols}")
    print(f"Target column: {target_col}")
    print(f"Initial dataframe shape: {df.shape}")
    
    # Verify all columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    total_samples = len(df) - lookback
    print(f"Total possible samples: {total_samples}")
    
    for i in range(total_samples):
        try:
            feature_seq = df[feature_cols].iloc[i:i + lookback].values
            target_val = df[target_col].iloc[i + lookback]
            
            # Verify sequence is complete
            if len(feature_seq) == lookback and not np.isnan(target_val):
                X.append(feature_seq)
                y.append(target_val)
            
            if i % 1000 == 0:  # Progress update
                print(f"Processed {i}/{total_samples} samples. Current X size: {len(X)}")
                
        except Exception as e:
            print(f"Error at index {i}: {str(e)}")
            continue
    
    print(f"\nTotal samples created: {len(X)}")
    
    if len(X) == 0:
        raise ValueError("No valid sequences created. Check data and parameters.")
    
    return np.array(X), np.array(y)

def build_model(lookback, n_features):
    """Build CNN model specifically for temporal patterns."""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(lookback, n_features)),
        
        # Capture short-term patterns (1-3 hour)
        layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Capture medium-term patterns (around 40-hour autocorrelation)
        layers.Conv1D(64, kernel_size=40, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Global patterns
        layers.GlobalAveragePooling1D(),
        
        # Regression layers
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1)  # Linear output for price prediction
    ])
    
    # Use a lower learning rate for fine-grained learning
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    
    # Use Huber loss for robustness to outliers
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.Huber(),
        metrics=['mae', 'mse']  # Track both mean absolute and squared errors
    )
    
    return model

def train_model():
    # 1. Fetch and prepare data
    df = fetch_historical_data(days=365)  # 1 year of data
    
    # 2. Define features
    feature_cols = [
        # Base features
        'close', 'volume', 'vwap', 'obv', 'mfi', 'ema', 'atr', 'vol_score',
        # Temporal features
        'vol_ratio_1', 'vol_ratio_2', 'vol_ratio_3',
        'vol_ratio_1_zscore', 'vol_ratio_2_zscore', 'vol_ratio_3_zscore',
        'price_mom_1', 'price_mom_3', 'price_mom_40',
        'vw_price_change',
        'volatility_3h', 'volatility_40h',
        'volume_ratio', 'volume_trend'  # New volume features
    ]
    
    # 3. Create sequences
    X, y = create_supervised_data(df, feature_cols, 'next_close', lookback=60)
    print("\nData shapes:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # 4. Scale features and target
    feature_scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_scaled = feature_scaler.fit_transform(X_reshaped)
    X = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Use RobustScaler for target to handle outliers
    target_scaler = RobustScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1))
    
    # 5. Train/test split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
    
    # 6. Build and train model
    model = build_model(60, len(feature_cols))
    print("\nModel summary:")
    model.summary()
    
    # Add callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001
    )
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # 7. Evaluate model
    print("\nEvaluating model...")
    
    # Get predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test_orig = target_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_test_orig))
    mse = np.mean((y_pred - y_test_orig) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"\nTest MAE: ${mae:.2f}")
    print(f"Test RMSE: ${rmse:.2f}")
    
    # Calculate directional accuracy
    direction_correct = np.sum((y_pred[1:] > y_pred[:-1]) == 
                             (y_test_orig[1:] > y_test_orig[:-1]))
    directional_accuracy = direction_correct / (len(y_pred) - 1)
    print(f"Directional Accuracy: {directional_accuracy:.2%}")
    
    # 8. Plot results
    plt.figure(figsize=(15, 5))
    
    # Training history
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Predictions vs Actual
    plt.subplot(1, 2, 2)
    plt.plot(y_test_orig[-100:], label='Actual')  # Last 100 points
    plt.plot(y_pred[-100:], label='Predicted')
    plt.title('Price Predictions (Last 100 Points)')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_v3.png')
    
    # 9. Save model and scalers
    model.save('jnj_model_v3.keras')
    np.save('feature_scaler.npy', [feature_scaler.mean_, feature_scaler.scale_])
    np.save('target_scaler.npy', [target_scaler.center_, target_scaler.scale_])
    print("\nModel and scalers saved")
    
    return model, history, (X_test, y_test_orig, y_pred)

if __name__ == "__main__":
    model, history, test_data = train_model()
