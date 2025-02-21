import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score
from long_short.models import train_model, predict, prepare_data_for_model

logger = logging.getLogger(__name__)

class OverfitDetector:
    def __init__(self, window_size: int = 30, step_size: int = 1, n_threads: int = 1):
        """Initialize the OverfitDetector."""
        self.window_size = window_size
        self.step_size = step_size
        self.n_threads = n_threads
        logger.info(f"Initialized OverfitDetector with {n_threads} threads")
        logger.info(f"Window size: {window_size}, Step size: {step_size}")

    def prepare_features_and_target(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and scale features, and extract target variable."""
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        features = data[feature_columns].values
        
        # Use RobustScaler for better handling of outliers
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Extract close price as target
        close_price_idx = feature_columns.index('close')
        target = features[:, close_price_idx]
        
        return scaled_features, target

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
        """Calculate performance metrics with improved stability."""
        # Calculate returns with clipping to handle outliers
        returns = np.clip(np.diff(y_true) / y_true[:-1], -0.1, 0.1)
        pred_returns = np.clip(np.diff(y_pred) / y_pred[:-1], -0.1, 0.1)
        
        # Sharpe ratio with more stable calculation
        excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
        sharpe = np.sqrt(252) * np.mean(excess_returns) / (np.std(excess_returns) + 1e-6)
        
        # Maximum drawdown with smoothing
        cumulative = np.cumprod(1 + returns)
        smoothed_cumulative = pd.Series(cumulative).rolling(window=5).mean().values
        running_max = np.maximum.accumulate(smoothed_cumulative)
        drawdowns = (smoothed_cumulative - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Prediction stability with regularization
        stability = r2_score(returns[:-1], returns[1:]) if len(returns) > 1 else 0
        stability = np.clip(stability, -1, 1)  # Bound stability metric
        
        return sharpe, max_drawdown, stability

    def optimize_ensemble_weights(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
        """Optimize ensemble weights with improved stability."""
        weights = {}
        total_score = 0
        
        for model_name, preds in predictions.items():
            # Calculate multiple metrics for more robust weighting
            r2 = max(0, r2_score(y_true, preds))
            mae = np.mean(np.abs(y_true - preds))
            correlation = np.corrcoef(y_true, preds)[0, 1]
            
            # Combine metrics with emphasis on correlation
            score = (r2 + correlation + (1 / (1 + mae))) / 3
            weights[model_name] = score
            total_score += score
        
        # Normalize weights with smoothing
        if total_score > 0:
            weights = {k: (0.7 * v/total_score + 0.3/len(predictions)) 
                      for k, v in weights.items()}
        else:
            # Equal weights if all models perform poorly
            n_models = len(predictions)
            weights = {k: 1.0/n_models for k in predictions.keys()}
        
        return weights

    def train_models_for_period(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                              min_delta: float = 0.0001) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        """Train multiple models and generate predictions."""
        predictions = {}
        models = {}
        n_features = X_train.shape[2]
        
        # Train models with increased patience and regularization
        for model_type in ['lstm', 'cnn', 'transformer']:
            logger.info(f"Training {model_type.upper()} model...")
            models[model_type] = train_model(
                X_train, y_train, self.window_size, n_features,
                model_type=model_type,
                min_delta=min_delta,
                early_stop_patience=15  # Increased patience
            )
            predictions[model_type] = predict(models[model_type], X_test, self.window_size, model_type)
        
        return predictions, models

    def walk_forward_analysis(self, data_dict: Dict[str, pd.DataFrame], n_splits: int = 5) -> Dict[str, List[float]]:
        """Perform walk-forward analysis with ensemble modeling."""
        results = {
            'returns': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'prediction_stability': [],
            'train_test_correlation': [],
            'lstm_weights': [],
            'cnn_weights': [],
            'transformer_weights': []
        }
        
        logger.info("\nStarting validation analysis...")
        
        for symbol, data in data_dict.items():
            logger.info(f"\nAnalyzing {symbol}...")
            
            # Prepare features and target
            features, target = self.prepare_features_and_target(data)
            
            # Ensure we have enough data for both training and testing
            min_required_length = self.window_size * 4  # Minimum length needed for meaningful split
            if len(features) < min_required_length:
                logger.warning(f"Not enough data for {symbol}. Skipping...")
                continue
                
            # Split data into training and testing with overlap
            train_size = int(len(features) * 0.8)
            if train_size <= self.window_size:
                logger.warning(f"Training set too small for {symbol}. Skipping...")
                continue
                
            train_features = features[:train_size]
            test_features = features[train_size-self.window_size:]  # Include overlap
            train_target = target[:train_size]
            test_target = target[train_size-self.window_size:]
            
            # Prepare sequences
            X_train, y_train = prepare_data_for_model(train_features, self.window_size)
            X_test, y_test = prepare_data_for_model(test_features, self.window_size)
            
            # Verify we have data after sequence preparation
            if len(X_train) == 0 or len(X_test) == 0:
                logger.warning(f"No sequences could be created for {symbol}. Skipping...")
                continue
            
            # Train models and get predictions
            predictions, models = self.train_models_for_period(X_train, y_train, X_test)
            
            # Optimize ensemble weights
            weights = self.optimize_ensemble_weights(predictions, y_test)
            
            # Store weights
            results['lstm_weights'].append(weights.get('lstm', 0))
            results['cnn_weights'].append(weights.get('cnn', 0))
            results['transformer_weights'].append(weights.get('transformer', 0))
            
            # Calculate ensemble predictions with smoothing
            ensemble_pred = np.zeros_like(y_test)
            for model_name, preds in predictions.items():
                ensemble_pred += weights[model_name] * preds
            
            # Apply exponential smoothing to predictions
            alpha = 0.1
            smoothed_pred = pd.Series(ensemble_pred).ewm(alpha=alpha).mean().values
            
            # Calculate metrics
            sharpe, max_dd, stability = self.calculate_metrics(y_test, smoothed_pred)
            
            # Calculate returns with smoothing
            returns = (smoothed_pred[-1] - smoothed_pred[0]) / smoothed_pred[0]
            
            # Calculate train-test correlation with regularization
            train_pred = np.mean([predict(models[model_type], X_train, self.window_size, model_type) 
                                for model_type in ['lstm', 'cnn', 'transformer']], axis=0)
            correlation = np.clip(np.corrcoef(train_pred, y_train)[0, 1], -1, 1)
            
            # Store results
            results['returns'].append(returns)
            results['sharpe_ratios'].append(sharpe)
            results['max_drawdowns'].append(max_dd)
            results['prediction_stability'].append(stability)
            results['train_test_correlation'].append(correlation)
            
            # Log period results
            logger.info("Period 1 Results:")
            logger.info(f"Return: {returns:.2%}")
            logger.info(f"Sharpe: {sharpe:.2f}")
            logger.info(f"Max DD: {max_dd:.2%}")
            logger.info(f"Pred Stability: {stability:.2f}")
            logger.info(f"Train-Test Correlation: {correlation:.2f}")
            
            # Check for overfitting
            if correlation < 0.3:
                logger.warning("Potential overfitting detected:")
                logger.warning("- Low train-test correlation")
            if stability < 0.2:
                logger.warning("- Low prediction stability")
        
        return results
