import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Concatenate, Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime

from momentum_ai_trading.utils.config import (
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    RESULTS_PATH
)

def create_sequences(data, seq_length=60):
    """Create sequences for LSTM training."""
    sequences = []
    targets = []
    
    for i in range(len(data) - seq_length):
        seq = data[i:(i + seq_length)]
        target = data.iloc[i + seq_length]
        
        # Calculate forward returns (next day's return)
        forward_return = (target['close'] - seq.iloc[-1]['close']) / seq.iloc[-1]['close']
        
        sequences.append(seq)
        targets.append(forward_return)
    
    return np.array(sequences), np.array(targets)

def build_deep_model(seq_length, n_features):
    """
    Build a deep learning model combining LSTM, CNN, and Attention mechanisms.
    """
    # Input layer
    inputs = Input(shape=(seq_length, n_features))
    
    # CNN branch for pattern recognition
    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    conv2 = Conv1D(filters=128, kernel_size=3, activation='relu')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv2)
    
    # LSTM branch for temporal dependencies
    lstm1 = LSTM(128, return_sequences=True)(inputs)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    
    # Attention mechanism
    attention = Attention()([lstm2, lstm2])
    
    # Combine CNN and LSTM branches
    concat = Concatenate()([pool1, attention])
    
    # Dense layers for final processing
    dense1 = Dense(128, activation='relu')(concat)
    bn1 = BatchNormalization()(dense1)
    dropout1 = Dropout(0.3)(bn1)
    
    dense2 = Dense(64, activation='relu')(dropout1)
    bn2 = BatchNormalization()(dense2)
    dropout2 = Dropout(0.2)(bn2)
    
    # Multiple output heads
    return_pred = Dense(1, name='return_prediction')(dropout2)  # Predict returns
    direction_pred = Dense(1, activation='sigmoid', name='direction_prediction')(dropout2)  # Predict direction
    
    model = Model(inputs=inputs, outputs=[return_pred, direction_pred])
    return model

def custom_loss(y_true, y_pred):
    """
    Custom loss function combining directional accuracy and return prediction.
    Puts more emphasis on getting the direction right.
    """
    # MSE for return prediction
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    
    # Direction accuracy loss
    direction_true = tf.sign(y_true)
    direction_pred = tf.sign(y_pred)
    direction_loss = tf.keras.losses.binary_crossentropy(
        (direction_true + 1) / 2,  # Scale to [0,1]
        (direction_pred + 1) / 2
    )
    
    # Combine losses with weights
    return 0.3 * mse_loss + 0.7 * direction_loss

def train_model(symbols=['AAPL', 'MSFT', 'GOOGL'], seq_length=60, epochs=100, batch_size=32):
    """
    Train deep learning model on multiple symbols.
    """
    print("Loading and preparing data...")
    all_sequences = []
    all_targets = []
    
    for symbol in symbols:
        # Load data
        data_path = os.path.join(PROCESSED_DATA_PATH, f"{symbol}_processed_daily.csv")
        data = pd.read_csv(data_path)
        
        # Calculate technical indicators
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        data['sma_200'] = data['close'].rolling(window=200).mean()
        
        data['rsi'] = calculate_rsi(data['close'])
        data['atr'] = calculate_atr(data['high'], data['low'], data['close'])
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = calculate_bollinger_bands(data['close'])
        
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        # Drop NaN values
        data = data.dropna()
        
        # Normalize features
        scaler = StandardScaler()
        features = ['open', 'high', 'low', 'close', 'volume', 
                   'sma_20', 'sma_50', 'sma_200', 'rsi', 'atr',
                   'bb_upper', 'bb_middle', 'bb_lower', 'volume_ratio']
        
        data_scaled = pd.DataFrame(scaler.fit_transform(data[features]), columns=features)
        
        # Save scaler for later use
        joblib.dump(scaler, f"{MODEL_PATH}_{symbol}_scaler.pkl")
        
        # Create sequences
        X, y = create_sequences(data_scaled, seq_length)
        
        all_sequences.append(X)
        all_targets.append(y)
    
    # Combine data from all symbols
    X = np.concatenate(all_sequences)
    y = np.concatenate(all_targets)
    
    # Split into train and validation sets
    train_size = int(len(X) * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Convert targets for direction prediction
    y_train_dir = np.where(y_train > 0, 1, 0)
    y_val_dir = np.where(y_val > 0, 1, 0)
    
    print("Building model...")
    model = build_deep_model(seq_length, len(features))
    
    # Compile model with custom loss
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'return_prediction': custom_loss,
            'direction_prediction': 'binary_crossentropy'
        },
        loss_weights={
            'return_prediction': 0.3,
            'direction_prediction': 0.7
        },
        metrics={
            'return_prediction': ['mae', 'mse'],
            'direction_prediction': ['accuracy']
        }
    )
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        f"{MODEL_PATH}_deep.h5",
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    print("Training model...")
    history = model.fit(
        X_train,
        {'return_prediction': y_train, 'direction_prediction': y_train_dir},
        validation_data=(
            X_val,
            {'return_prediction': y_val, 'direction_prediction': y_val_dir}
        ),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=1
    )
    
    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(RESULTS_PATH, 'deep_model_history.csv'))
    
    # Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(
        X_val,
        {'return_prediction': y_val, 'direction_prediction': y_val_dir},
        verbose=1
    )
    
    print("\nValidation Results:")
    for metric, value in zip(model.metrics_names, results):
        print(f"{metric}: {value:.4f}")
    
    # Save model architecture
    model_json = model.to_json()
    with open(f"{MODEL_PATH}_deep_architecture.json", "w") as json_file:
        json_file.write(model_json)
    
    return model, history

def calculate_rsi(prices, period=14):
    """Calculate RSI technical indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range (ATR)."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_bollinger_bands(prices, period=20, num_std=2):
    """Calculate Bollinger Bands."""
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train model
    model, history = train_model()
