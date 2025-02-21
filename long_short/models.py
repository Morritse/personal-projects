import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LayerNormalization, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import numpy as np
from typing import Tuple
import logging
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

def prepare_data_for_model(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for model training by creating sequences and targets.
    
    Args:
        data: Input data array
        window_size: Size of the sliding window
        
    Returns:
        Tuple of (X, y) where X is the input sequences and y is the target values
    """
    X = []
    y = []
    
    # Get close price index (assuming it's the 4th column - index 3)
    close_idx = 3
    
    for i in range(len(data) - window_size):
        # Use all features for input sequence
        X.append(data[i:(i + window_size)])
        # Use only close price for target
        y.append(data[i + window_size, close_idx])
        
    return np.array(X), np.array(y)

def create_lstm_model(window_size: int, n_features: int) -> Sequential:
    """Create LSTM model with improved architecture and regularization."""
    model = Sequential([
        # Input layer with batch normalization
        BatchNormalization(input_shape=(window_size, n_features)),
        
        # First LSTM layer
        LSTM(128, return_sequences=True,
             kernel_regularizer=l2(0.01),
             recurrent_regularizer=l2(0.01)),
        Dropout(0.3),
        BatchNormalization(),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True,
             kernel_regularizer=l2(0.01),
             recurrent_regularizer=l2(0.01)),
        Dropout(0.3),
        BatchNormalization(),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False,
             kernel_regularizer=l2(0.01),
             recurrent_regularizer=l2(0.01)),
        Dropout(0.3),
        BatchNormalization(),
        
        # Dense layers
        Dense(32, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(16, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    return model

def create_cnn_model(window_size: int, n_features: int) -> Sequential:
    """Create CNN model with improved architecture and regularization."""
    model = Sequential([
        # Input layer with batch normalization
        BatchNormalization(input_shape=(window_size, n_features)),
        
        # First convolution block
        Conv1D(filters=128, kernel_size=3, activation='relu',
               kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Second convolution block
        Conv1D(filters=64, kernel_size=3, activation='relu',
               kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Third convolution block
        Conv1D(filters=32, kernel_size=3, activation='relu',
               kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Dense layers
        Flatten(),
        Dense(32, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(16, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    return model

def create_transformer_model(window_size: int, n_features: int) -> Sequential:
    """Create improved Transformer-inspired model with better regularization."""
    model = Sequential([
        # Input processing
        BatchNormalization(input_shape=(window_size, n_features)),
        Conv1D(filters=64, kernel_size=1, activation='relu',
               kernel_regularizer=l2(0.01)),
        LayerNormalization(epsilon=1e-6),
        
        # First temporal feature extraction block
        Conv1D(filters=128, kernel_size=3, activation='relu',
               padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        LayerNormalization(epsilon=1e-6),
        
        # Second temporal feature extraction block
        Conv1D(filters=64, kernel_size=3, activation='relu',
               padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        LayerNormalization(epsilon=1e-6),
        
        # Third temporal feature extraction block
        Conv1D(filters=32, kernel_size=3, activation='relu',
               padding='same', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output processing
        Flatten(),
        Dense(64, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dropout(0.3),
        Dense(32, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(16, activation='relu',
              kernel_regularizer=l2(0.01)),
        Dense(1)
    ])
    return model

def train_model(X_train: np.ndarray, 
                y_train: np.ndarray,
                window_size: int,
                n_features: int,
                epochs: int = 100,
                batch_size: int = 32,
                model_type: str = 'lstm',
                early_stop_patience: int = 15,
                min_delta: float = 0.0001) -> Sequential:
    """
    Train a deep learning model with improved training process.
    
    Args:
        X_train: Training features
        y_train: Training targets
        window_size: Size of the sliding window
        n_features: Number of input features
        epochs: Number of training epochs
        batch_size: Batch size for training
        model_type: Type of model ('lstm', 'cnn', or 'transformer')
        early_stop_patience: Number of epochs with no improvement after which training will be stopped
        min_delta: Minimum change in monitored quantity to qualify as an improvement
        
    Returns:
        Trained model
    """
    if model_type == 'lstm':
        model = create_lstm_model(window_size, n_features)
    elif model_type == 'cnn':
        model = create_cnn_model(window_size, n_features)
    elif model_type == 'transformer':
        model = create_transformer_model(window_size, n_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Learning rate schedule with warm-up and decay
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate,
        first_decay_steps=1000,
        t_mul=2.0,
        m_mul=0.9,
        alpha=0.1
    )
    optimizer = Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # More robust to outliers than MSE
        metrics=['mae']
    )
    
    # Early stopping with restoration of best weights
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=early_stop_patience,
        min_delta=min_delta,
        restore_best_weights=True,
        mode='min'
    )
    
    # Train model with validation split
    validation_split = 0.2
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model

def predict(model: Sequential,
           X: np.ndarray,
           window_size: int,
           model_type: str) -> np.ndarray:
    """Make predictions with smoothing."""
    predictions = model.predict(X, verbose=0)
    return predictions.flatten()
