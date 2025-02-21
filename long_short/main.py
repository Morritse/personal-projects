import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import tensorflow as tf
from pathlib import Path
import os
import logging
from datetime import datetime
from long_short.utils import extract_features, create_sliding_windows
from long_short.models import train_model, predict, prepare_data_for_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate the Sharpe ratio for a series of returns."""
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    excess_returns = returns - daily_rf
    if len(excess_returns) < 2:
        return 0
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate the maximum drawdown from a series of returns."""
    cumulative = (1 + returns).cumprod()
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative / rolling_max - 1
    return np.min(drawdowns)

def construct_portfolio(data_dict: Dict[str, pd.DataFrame], 
                       window_size: int = 120,
                       step_size: int = 1,
                       model_types: List[str] = ['lstm', 'cnn', 'transformer']) -> Tuple[float, float, float]:
    """
    Construct a long-short portfolio using an ensemble of models.
    
    Args:
        data_dict: Dictionary mapping stock symbols to their DataFrame of features
        window_size: Size of the sliding window for sequence creation
        step_size: Step size for sliding windows
        model_types: List of model types to use in ensemble
    
    Returns:
        Tuple of (total_return, sharpe_ratio, max_drawdown)
    """
    logger.info("\nConstructing portfolio...")
    
    # Store predictions for each stock
    all_predictions = {}
    
    # Training parameters
    epochs = 200  # Increased epochs for better convergence
    batch_size = 64  # Larger batch size for stability
    
    # Train models and make predictions for each stock
    for symbol, data in data_dict.items():
        logger.info(f"\nProcessing {symbol}...")
        logger.info(f"Data shape: {data.shape}")
        
        try:
            # Extract features
            features = extract_features(data)
            logger.info(f"Features extracted: {features.columns.tolist()}")
            
            X = features.drop(columns=['target']).values
            y = features['returns'].values
            
            # Create sliding windows
            logger.info(f"Creating sequences with window_size={window_size}, step_size={step_size}")
            X_windows = create_sliding_windows(X, window_size, step_size)
            y_windows = create_sliding_windows(y, window_size, step_size)
            logger.info(f"Sequence shape: {X_windows.shape}")
            
            # Prepare data for model
            X_train, X_test, y_train, y_test, scaler = prepare_data_for_model(X_windows, window_size, step_size)
            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Test data shape: {X_test.shape}")
            
            # Train models and get predictions
            model_predictions = []
            for model_type in model_types:
                logger.info(f"\nTraining {model_type.upper()} model...")
                model = train_model(
                    X_train, y_train, 
                    window_size, 
                    X_train.shape[2],
                    epochs=epochs,
                    batch_size=batch_size,
                    model_type=model_type
                )
                preds = predict(model, X_test, window_size, model_type)
                model_predictions.append(preds)
                logger.info(f"{model_type.upper()} training complete")
            
            # Ensemble predictions (simple average)
            all_predictions[symbol] = np.mean(model_predictions, axis=0)
            logger.info(f"Ensemble predictions shape for {symbol}: {all_predictions[symbol].shape}")
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue
    
    if not all_predictions:
        raise ValueError("No predictions generated for any symbols")
    
    # Construct portfolio returns
    logger.info("\nConstructing portfolio returns...")
    portfolio_returns = []
    
    # Get the minimum length of predictions across all stocks
    min_length = min(len(results) for results in all_predictions.values())
    
    # For each time step
    for t in range(min_length):
        # Get predictions for all stocks at time t
        stock_preds = {symbol: preds[t] for symbol, preds in all_predictions.items()}
        
        # Split into long and short positions
        long_positions = {s: p for s, p in stock_preds.items() if p > 0}
        short_positions = {s: p for s, p in stock_preds.items() if p <= 0}
        
        # Calculate weights (equal weighting)
        long_weight = 1.0 / len(long_positions) if long_positions else 0
        short_weight = 1.0 / len(short_positions) if short_positions else 0
        
        # Calculate portfolio return for this time step
        port_return = 0
        
        # Add long positions
        for symbol in long_positions:
            actual_return = data_dict[symbol]['returns'].iloc[t + window_size]
            port_return += long_weight * actual_return
            
        # Add short positions
        for symbol in short_positions:
            actual_return = data_dict[symbol]['returns'].iloc[t + window_size]
            port_return -= short_weight * actual_return  # Negative sign for short positions
            
        portfolio_returns.append(port_return)
    
    # Convert to numpy array
    portfolio_returns = np.array(portfolio_returns)
    
    # Calculate metrics
    logger.info("\nCalculating portfolio metrics...")
    total_return = np.prod(1 + portfolio_returns) - 1
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
    max_drawdown = calculate_max_drawdown(portfolio_returns)
    
    logger.info(f"Total Return: {total_return:.2%}")
    logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {max_drawdown:.2%}")
    
    return total_return, sharpe_ratio, max_drawdown

def run_backtest(symbols: List[str], window_size: int = 120, step_size: int = 1) -> Dict:
    """
    Run backtest for a list of symbols.
    
    Args:
        symbols: List of stock symbols to include in portfolio
        window_size: Size of the sliding window
        step_size: Step size for sliding windows
    
    Returns:
        Dictionary containing backtest results
    """
    logger.info("\nStarting backtest...")
    logger.info(f"Loading data for symbols: {symbols}")
    
    # Load data for all symbols
    data_dict = {}
    for symbol in symbols:
        try:
            filepath = f'momentum_ai_trading/data/processed/{symbol}_processed_daily.csv'
            logger.info(f"Loading data from: {filepath}")
            
            if not os.path.exists(filepath):
                logger.warning(f"Data file not found for {symbol}")
                continue
                
            data = pd.read_csv(filepath)
            logger.info(f"Loaded {len(data)} rows for {symbol}")
            data_dict[symbol] = data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            continue
    
    if not data_dict:
        raise ValueError("No data loaded for any symbols")
    
    # Run portfolio construction
    total_return, sharpe_ratio, max_drawdown = construct_portfolio(
        data_dict,
        window_size=window_size,
        step_size=step_size
    )
    
    # Return results
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'symbols': list(data_dict.keys())
    }
