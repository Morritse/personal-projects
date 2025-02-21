import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization,
    Conv1D, LayerNormalization, Activation, Layer, Add, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import numpy as np
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StabilityMonitorCallback(Callback):
    """Monitor numerical stability during training."""
    def __init__(self):
        super().__init__()
        self.nan_count = 0
        self.max_nans = 5  # Maximum number of NaN batches before stopping
        
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        # Check for NaN/Inf values
        for metric, value in logs.items():
            if np.isnan(value) or np.isinf(value):
                self.nan_count += 1
                logger.warning(f"❌ {metric} has {value} at batch {batch}")
                if self.nan_count >= self.max_nans:
                    logger.error("Too many NaN values, stopping training")
                    self.model.stop_training = True
                    return
                
                # Log recent values for debugging
                if hasattr(self.model, 'history') and self.model.history is not None:
                    logger.info(f"Recent values: {self.model.history.history.get(metric, [])[-5:]}")
        
        # Reset counter if batch was good
        if all(not (np.isnan(value) or np.isinf(value)) for value in logs.values()):
            self.nan_count = 0

def custom_return_loss(y_true, y_pred):
    """Stable return prediction loss."""
    epsilon = 1e-7
    
    # Ensure inputs are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Clip values
    y_pred = tf.clip_by_value(y_pred, -0.1, 0.1)
    y_true = tf.clip_by_value(y_true, -0.1, 0.1)
    
    # Use Huber loss for robustness
    huber = tf.keras.losses.Huber(delta=0.1)
    loss = huber(y_true, y_pred)
    
    # Check for NaN/Inf
    if tf.reduce_any(tf.math.is_nan(loss)) or tf.reduce_any(tf.math.is_inf(loss)):
        tf.print("\nWarning: NaN/Inf in return loss", loss)
        return tf.constant(0.1, dtype=tf.float32)  # Fallback value
    
    return loss

def custom_direction_loss(y_true, y_pred):
    """Stable direction prediction loss."""
    epsilon = 1e-7
    
    # Convert returns to direction and ensure float32
    y_true_dir = tf.cast(y_true, tf.float32)  # Already binary
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Ensure predictions are between epsilon and 1-epsilon
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Binary cross entropy with stability
    bce = -(y_true_dir * tf.math.log(y_pred + epsilon) +
            (1 - y_true_dir) * tf.math.log(1 - y_pred + epsilon))
    loss = tf.reduce_mean(bce)
    
    # Check for NaN/Inf
    if tf.reduce_any(tf.math.is_nan(loss)) or tf.reduce_any(tf.math.is_inf(loss)):
        tf.print("\nWarning: NaN/Inf in direction loss", loss)
        return tf.constant(0.1, dtype=tf.float32)  # Fallback value
    
    return loss

class DeepTradingModel:
    def __init__(self, seq_length, n_features):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = self.build_model()
        
    def build_model(self):
        """Build model optimized for stability."""
        # Input layer with batch normalization
        inputs = Input(shape=(self.seq_length, self.n_features))
        x = BatchNormalization()(inputs)
        
        # Project input to match CNN output dimension
        x_proj = Conv1D(128, 1, padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        x_proj = LayerNormalization()(x_proj)
        
        # LSTM branch with layer normalization
        lstm = LSTM(128, return_sequences=True, 
                   recurrent_initializer='glorot_uniform',
                   kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                   recurrent_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        lstm = LayerNormalization()(lstm)
        lstm = Dropout(0.2)(lstm)
        
        lstm2 = LSTM(128, return_sequences=True,
                    recurrent_initializer='glorot_uniform',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-5),
                    recurrent_regularizer=tf.keras.regularizers.l2(1e-5))(lstm)
        lstm2 = LayerNormalization()(lstm2)
        lstm2 = Dropout(0.2)(lstm2)
        
        # CNN branch with residual connections
        conv1 = Conv1D(128, 3, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
        conv1 = LayerNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv1 = Add()([conv1, x_proj])  # Residual connection with projected input
        
        conv2 = Conv1D(128, 3, padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(conv1)
        conv2 = LayerNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        conv2 = Add()([conv2, conv1])  # Residual connection
        
        # Multi-head attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=32
        )(lstm2, lstm2)
        attention = LayerNormalization()(attention)
        
        # Combine branches
        concat = Concatenate()([conv2, attention])
        
        # Global features
        pooled = tf.keras.layers.GlobalAveragePooling1D()(concat)
        
        # Dense layers with batch normalization
        dense1 = Dense(256, kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(pooled)
        dense1 = BatchNormalization()(dense1)
        dense1 = Activation('relu')(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(128, kernel_initializer='he_normal',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Activation('relu')(dense2)
        dense2 = Dropout(0.2)(dense2)
        
        # Output heads with stability
        return_pred = Dense(1, name='return_prediction',
                          kernel_initializer='glorot_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-5))(dense2)
        direction_pred = Dense(1, activation='sigmoid', name='direction_prediction',
                             kernel_initializer='glorot_normal',
                             kernel_regularizer=tf.keras.regularizers.l2(1e-5))(dense2)
        
        model = Model(inputs=inputs, outputs=[return_pred, direction_pred])
        
        # Compile with stability improvements
        optimizer = Adam(learning_rate=0.0001, clipnorm=0.5)
        
        model.compile(
            optimizer=optimizer,
            loss={
                'return_prediction': custom_return_loss,
                'direction_prediction': custom_direction_loss
            },
            loss_weights={
                'return_prediction': 0.3,
                'direction_prediction': 0.7
            },
            metrics={
                'return_prediction': [
                    tf.keras.metrics.MeanAbsoluteError(name='mae'),
                    tf.keras.metrics.MeanSquaredError(name='mse')
                ],
                'direction_prediction': [
                    tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5)
                ]
            }
        )
        
        return model
    
    def get_callbacks(self):
        """Get training callbacks with stability monitoring."""
        # Create logs directory if it doesn't exist
        log_dir = os.path.join('data', 'deep_models', 'logs', 
                              datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        
        return [
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join('data', 'deep_models', 'best_model.keras'),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            StabilityMonitorCallback(),
            tf.keras.callbacks.TensorBoard(
                log_dir=log_dir,
                histogram_freq=1,
                update_freq='batch'
            )
        ]
    
    def fit(self, train_dataset, val_dataset, epochs=50):
        """Train model with stability monitoring."""
        # Set smaller batch size
        batch_size = 16  # Reduced from 32
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = val_dataset.batch(batch_size)
        
        # Train with monitoring
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        return history
