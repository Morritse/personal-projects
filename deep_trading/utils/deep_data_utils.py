import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, RobustScaler
import logging

logger = logging.getLogger(__name__)

def validate_features(df, name=""):
    """Validate features for numerical stability."""
    issues = []
    
    # Check for NaN/Inf
    if df.isna().any().any():
        nan_cols = df.columns[df.isna().any()].tolist()
        issues.append(f"NaN values found in columns: {nan_cols}")
    
    if (df.abs() == np.inf).any().any():
        inf_cols = df.columns[(df.abs() == np.inf).any()].tolist()
        issues.append(f"Infinite values found in columns: {inf_cols}")
    
    # Check for extreme values
    for col in df.select_dtypes(include=[np.number]).columns:
        stats = df[col].describe()
        iqr = stats['75%'] - stats['25%']
        lower_bound = stats['25%'] - 3 * iqr
        upper_bound = stats['75%'] + 3 * iqr
        
        outliers = df[col][(df[col] < lower_bound) | (df[col] > upper_bound)]
        if len(outliers) > 0:
            issues.append(f"Column {col} has {len(outliers)} outliers")
    
    if issues:
        logger.warning(f"\nData validation issues in {name}:")
        for issue in issues:
            logger.warning(f"- {issue}")
        return False
    
    return True

def clean_features(df):
    """Clean and stabilize features."""
    # Replace inf with large numbers
    df = df.replace([np.inf, -np.inf], [1e6, -1e6])
    
    # Fill NaN with forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Clip extreme values
    for col in df.select_dtypes(include=[np.number]).columns:
        stats = df[col].describe()
        iqr = stats['75%'] - stats['25%']
        lower_bound = stats['25%'] - 5 * iqr
        upper_bound = stats['75%'] + 5 * iqr
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def prepare_features_for_prediction(data, seq_length=60):
    """Prepare features with stability checks."""
    try:
        logger.info("Starting feature preparation...")
        
        # Use RobustScaler for better outlier handling
        scaler = RobustScaler()
        
        # Separate numerical and categorical columns
        datetime_cols = ['datetime']
        categorical_cols = ['symbol', 'instrument_type', 'market_regime', 'tech_regime', 
                          'volatility_regime', 'bear_regime', 'rates_regime']
        numerical_cols = [col for col in data.columns 
                         if col not in datetime_cols + categorical_cols]
        
        # Scale numerical features
        numerical_data = pd.DataFrame(
            scaler.fit_transform(data[numerical_cols].astype(float)),
            columns=numerical_cols,
            index=data.index
        )
        
        # Process categorical features
        categorical_data = pd.DataFrame(index=data.index)
        for col in categorical_cols:
            if col in data.columns:
                # One-hot encode
                dummies = pd.get_dummies(data[col], prefix=col)
                categorical_data = pd.concat([categorical_data, dummies], axis=1)
        
        # Combine features
        processed_data = pd.concat([numerical_data, categorical_data], axis=1)
        
        # Convert to float32 for better numerical stability
        tensor_data = tf.cast(processed_data.values, tf.float32)
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(len(tensor_data) - seq_length):
            seq = tensor_data[i:(i + seq_length)]
            target = tensor_data[i + seq_length]
            
            # Skip sequence if it contains any NaN or Inf
            if tf.reduce_any(tf.math.is_nan(seq)) or tf.reduce_any(tf.math.is_inf(seq)):
                continue
                
            sequences.append(seq)
            targets.append(target)
        
        if not sequences:
            raise ValueError("No valid sequences created")
        
        # Convert to tensors
        X = tf.stack(sequences)
        y = tf.stack(targets)
        
        # Calculate returns (using Close price index)
        close_idx = numerical_cols.index('Close')
        returns = (y[:, close_idx] - X[:, -1, close_idx]) / (X[:, -1, close_idx] + 1e-7)
        
        # Clip returns to prevent extreme values
        returns = tf.clip_by_value(returns, -0.1, 0.1)
        
        # Create direction labels
        directions = tf.cast(returns > 0, tf.float32)
        
        # Split train/val
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        returns_train, returns_val = returns[:train_size], returns[train_size:]
        directions_train, directions_val = directions[:train_size], directions[train_size:]
        
        # Create datasets with prefetch
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (X_train, {
                'return_prediction': returns_train,
                'direction_prediction': directions_train
            })
        )
        
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (X_val, {
                'return_prediction': returns_val,
                'direction_prediction': directions_val
            })
        )
        
        logger.info(f"Created {len(sequences)} valid sequences")
        logger.info(f"Feature dimension: {X.shape[2]}")
        
        return train_dataset, val_dataset, X.shape[2]
        
    except Exception as e:
        logger.error(f"Error in feature preparation: {str(e)}")
        logger.error("Full traceback:", exc_info=True)
        raise
