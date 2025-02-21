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
    
    # Calculate price changes
    df['price_change'] = df['close'].diff()
    df['price_change_pct'] = df['close'].pct_change()
    df['next_price_change'] = df['price_change'].shift(-1)  # Target variable
    df['next_price_change_pct'] = df['price_change_pct'].shift(-1)
    
    # Technical indicators
    df['vwap'] = strategy.get_vwap(df)
    df['obv'] = strategy.get_obv(df)
    df['mfi'] = strategy.get_mfi(df)
    df['ema'] = strategy.get_ema(df)
    df['atr'] = strategy.get_atr(df)
    df['vol_score'] = strategy.get_volume_pattern_score(df)
    
    # Price relative to indicators
    df['close_vwap_ratio'] = df['close'] / df['vwap']
    df['close_ema_ratio'] = df['close'] / df['ema']
    
    # Volume patterns
    df['volume_ma3'] = df['volume'].rolling(3).mean()
    df['volume_ma40'] = df['volume'].rolling(40).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma3']
    df['volume_trend'] = df['volume_ma3'] / df['volume_ma40']
    
    # Momentum features
    for lag in [1, 3, 40]:  # Including the lag-40 autocorrelation
        df[f'price_mom_{lag}'] = df['close'].pct_change(lag)
        df[f'volume_mom_{lag}'] = df['volume'].pct_change(lag)
    
    # Volatility features
    df['volatility_3h'] = df['price_change_pct'].rolling(3).std()
    df['volatility_40h'] = df['price_change_pct'].rolling(40).std()
    
    # Print distribution info
    print("\nPrice change distribution:")
    print(df['price_change'].describe())
    print("\nNext price change distribution:")
    print(df['next_price_change'].describe())
    
    # Handle NaN values
    # First, forward fill indicators that should maintain last value
    indicators_ffill = ['ema', 'obv', 'vwap']
    df[indicators_ffill] = df[indicators_ffill].fillna(method='ffill')
    
    # For rolling calculations, we need a minimum window
    min_window = 100  # Maximum of our rolling windows
    df = df.iloc[min_window:]
    
    # Now drop any remaining NaN values
    df = df.dropna()
    
    print("\nFinal row count:", len(df))
    
    return df

def create_supervised_data(df, feature_cols, target_col='next_price_change', lookback=60):
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
        feature_seq = df[feature_cols].iloc[i:i + lookback].values
        target_val = df[target_col].iloc[i + lookback]
        
        # Verify sequence is complete
        if len(feature_seq) == lookback and not np.isnan(target_val):
            X.append(feature_seq)
            y.append(target_val)
        
        if i % 1000 == 0:  # Progress update
            print(f"Processed {i}/{total_samples} samples. Current X size: {len(X)}")
    
    print(f"\nTotal samples created: {len(X)}")
    
    if len(X) == 0:
        raise ValueError("No valid sequences created. Check data and parameters.")
    
    return np.array(X), np.array(y)

def build_model(lookback, n_features):
    """Build CNN model for price change prediction."""
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(lookback, n_features)),
        
        # First Conv1D block - short-term patterns
        layers.Conv1D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Second Conv1D block - medium-term patterns
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),
        
        # Global patterns
        layers.GlobalAveragePooling1D(),
        
        # Dense layers
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(1)  # Linear output for price change prediction
    ])
    
    # Use Huber loss for robustness to outliers
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.Huber(),
        metrics=['mae', 'mse']
    )
    
    return model

def train_model():
    # 1. Fetch and prepare data
    df = fetch_historical_data(days=365*3)  # 3 years of data
    
    # 2. Define features
    feature_cols = [
        # Price and volume
        'price_change', 'price_change_pct', 'volume_ratio',
        
        # Technical indicators
        'close_vwap_ratio', 'close_ema_ratio', 'mfi', 'atr', 'vol_score',
        
        # Momentum
        'price_mom_1', 'price_mom_3', 'price_mom_40',
        'volume_mom_1', 'volume_mom_3', 'volume_mom_40',
        
        # Volatility
        'volatility_3h', 'volatility_40h'
    ]
    
    # 3. Create sequences
    X, y = create_supervised_data(df, feature_cols, 'next_price_change', lookback=60)
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
    
    # Train model with larger batch size for more data
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,  # Increased batch size for more data
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
    direction_correct = np.sum((y_pred[1:] > 0) == (y_test_orig[1:] > 0))
    directional_accuracy = direction_correct / (len(y_pred) - 1)
    print(f"Directional Accuracy: {directional_accuracy:.2%}")
    
    # Calculate profit potential (assuming perfect execution)
    correct_direction_gains = np.sum(np.abs(y_test_orig[1:]) * (y_pred[1:] > 0) == (y_test_orig[1:] > 0))
    average_gain_per_trade = correct_direction_gains / (len(y_pred) - 1)
    print(f"Average Gain Per Trade: ${average_gain_per_trade:.2f}")
    
    # Calculate win rate by trade size
    small_moves = np.abs(y_test_orig) <= np.percentile(np.abs(y_test_orig), 50)
    large_moves = np.abs(y_test_orig) > np.percentile(np.abs(y_test_orig), 50)
    
    small_moves_accuracy = np.mean((y_pred[small_moves] > 0) == (y_test_orig[small_moves] > 0))
    large_moves_accuracy = np.mean((y_pred[large_moves] > 0) == (y_test_orig[large_moves] > 0))
    
    print(f"\nDirectional Accuracy by Move Size:")
    print(f"Small Moves (≤${np.percentile(np.abs(y_test_orig), 50):.2f}): {small_moves_accuracy:.2%}")
    print(f"Large Moves (>${np.percentile(np.abs(y_test_orig), 50):.2f}): {large_moves_accuracy:.2%}")
    
    # 8. Plot results
    plt.figure(figsize=(15, 10))
    
    # Training history
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Predictions vs Actual
    plt.subplot(2, 1, 2)
    plt.plot(y_test_orig[-100:], label='Actual Price Change')
    plt.plot(y_pred[-100:], label='Predicted Price Change')
    plt.title('Price Change Predictions (Last 100 Points)')
    plt.xlabel('Time')
    plt.ylabel('Price Change ($)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_v4.png')
    
    # 9. Save model and scalers
    model.save('jnj_model_v4.keras')
    np.save('feature_scaler_v4.npy', [feature_scaler.mean_, feature_scaler.scale_])
    np.save('target_scaler_v4.npy', [target_scaler.center_, target_scaler.scale_])
    print("\nModel and scalers saved")
    
    return model, history, (X_test, y_test_orig, y_pred)

if __name__ == "__main__":
    model, history, test_data = train_model()
