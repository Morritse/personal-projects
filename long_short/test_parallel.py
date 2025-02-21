import logging
from long_short.validation import OverfitDetector
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_model_stability(results: dict) -> None:
    """Analyze model stability and overfitting indicators with improved metrics."""
    logger.info("\nModel Stability Analysis:")
    logger.info("=======================")
    
    # Analyze return consistency with outlier handling
    returns = np.array(results['returns'])
    returns = returns[~np.isnan(returns)]  # Remove NaN values
    if len(returns) > 0:
        # Use median absolute deviation for more robust measure
        mad = np.median(np.abs(returns - np.median(returns)))
        logger.info(f"Return MAD: {mad:.2%}")
        if mad > 0.05:  # Lower threshold for MAD
            logger.warning("High variability in returns - possible overfitting")
    
    # Analyze Sharpe ratio stability with outlier handling
    sharpes = np.array(results['sharpe_ratios'])
    sharpes = sharpes[~np.isnan(sharpes)]  # Remove NaN values
    if len(sharpes) > 0:
        sharpe_std = np.std(sharpes)
        logger.info(f"Sharpe Ratio Std: {sharpe_std:.2f}")
        if sharpe_std > 0.5:
            logger.warning("High variability in Sharpe ratios - possible overfitting")
    
    # Analyze prediction stability with improved metric
    stabilities = np.array(results['prediction_stability'])
    stabilities = stabilities[~np.isnan(stabilities)]  # Remove NaN values
    if len(stabilities) > 0:
        mean_stability = np.mean(stabilities)
        logger.info(f"Mean Prediction Stability: {mean_stability:.2f}")
        if mean_stability < -0.5:  # Adjusted threshold
            logger.warning("Low prediction stability - possible overfitting")
    
    # Analyze train-test correlation
    correlations = np.array(results['train_test_correlation'])
    correlations = correlations[~np.isnan(correlations)]  # Remove NaN values
    if len(correlations) > 0:
        mean_correlation = np.mean(correlations)
        logger.info(f"Mean Train-Test Correlation: {mean_correlation:.2f}")
        if mean_correlation < 0.5:  # Adjusted threshold
            logger.warning("Low train-test correlation - likely overfitting")

def analyze_ensemble_weights(results: dict) -> None:
    """Analyze ensemble model weights and contributions with improved metrics."""
    logger.info("\nEnsemble Analysis:")
    logger.info("=================")
    
    # Calculate average model contributions with smoothing
    model_weights = {
        'LSTM': np.mean([w for w in results.get('lstm_weights', []) if not np.isnan(w)]),
        'CNN': np.mean([w for w in results.get('cnn_weights', []) if not np.isnan(w)]),
        'Transformer': np.mean([w for w in results.get('transformer_weights', []) if not np.isnan(w)])
    }
    
    # Normalize weights to ensure they sum to 1
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
    
    # Add minimum weight floor
    min_weight = 0.1
    for model, weight in model_weights.items():
        if weight < min_weight:
            logger.warning(f"Low contribution from {model} - potential model ineffectiveness")
        logger.info(f"{model} contribution: {weight:.2%}")
        if weight > 0.5:
            logger.warning(f"High dependence on {model} - potential instability")

def test_parallel_training():
    """Test parallel training with cross-validation and ensemble analysis."""
    start_time = time.time()
    
    logger.info("\nTesting Parallel Training")
    logger.info("=======================")
    
    # Load data with error handling
    symbol = 'AAPL'
    filepath = f'momentum_ai_trading/data/processed/{symbol}_processed_daily.csv'
    logger.info(f"\nLoading data for {symbol}...")
    
    try:
        data = pd.read_csv(filepath)
        data = data.tail(250)  # Use last year of data
        logger.info(f"Data shape: {data.shape}")
        
        if data.empty:
            logger.error("No data loaded")
            return
            
        # Check for required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            logger.error("Missing required columns")
            return
            
        # Remove any rows with NaN values
        data = data.dropna()
        if len(data) < 100:  # Minimum required length
            logger.error("Insufficient data after cleaning")
            return
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return
    
    # Initialize detector with improved parameters
    detector = OverfitDetector(
        window_size=30,  # Balanced window size
        step_size=1,
        n_threads=1
    )
    
    # Perform time series cross-validation with overlap
    tscv = TimeSeriesSplit(n_splits=5, test_size=50)  # Fixed test size
    all_results = []
    
    for train_index, test_index in tscv.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        
        # Ensure minimum data requirements
        if len(train_data) < detector.window_size * 3:
            logger.warning("Insufficient training data for fold")
            continue
            
        if len(test_data) < detector.window_size * 2:
            logger.warning("Insufficient test data for fold")
            continue
        
        # Create data dictionary for validation
        data_dict = {symbol: pd.concat([train_data, test_data])}
        
        # Run validation
        try:
            results = detector.walk_forward_analysis(data_dict, n_splits=2)
            if any(len(v) > 0 for v in results.values()):  # Only append if we got valid results
                all_results.append(results)
        except Exception as e:
            logger.error(f"Error in validation: {str(e)}")
            continue
    
    if not all_results:
        logger.warning("No valid results obtained from any fold")
        return
    
    # Aggregate results, handling empty lists and NaN values
    aggregated_results = {
        'returns': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'prediction_stability': [],
        'train_test_correlation': [],
        'lstm_weights': [],
        'cnn_weights': [],
        'transformer_weights': []
    }
    
    for result in all_results:
        for key in aggregated_results:
            values = result[key]
            if values:  # Only append if there are values
                # Filter out NaN values
                valid_values = [v for v in values if not np.isnan(v)]
                aggregated_results[key].extend(valid_values)
    
    # Analyze results with improved metrics
    logger.info("\nCross-Validation Results:")
    logger.info("========================")
    
    # Calculate metrics with NaN handling
    if aggregated_results['returns']:
        returns = np.array([r for r in aggregated_results['returns'] if not np.isnan(r)])
        if len(returns) > 0:
            logger.info(f"Mean Return: {np.mean(returns):.2%}")
            
    if aggregated_results['sharpe_ratios']:
        sharpes = np.array([s for s in aggregated_results['sharpe_ratios'] if not np.isnan(s)])
        if len(sharpes) > 0:
            logger.info(f"Mean Sharpe: {np.mean(sharpes):.2f}")
            
    if aggregated_results['max_drawdowns']:
        drawdowns = np.array([d for d in aggregated_results['max_drawdowns'] if not np.isnan(d)])
        if len(drawdowns) > 0:
            logger.info(f"Mean Max DD: {np.mean(drawdowns):.2%}")
    
    # Detailed analysis
    analyze_model_stability(aggregated_results)
    analyze_ensemble_weights(aggregated_results)
    
    # Execution time
    execution_time = time.time() - start_time
    logger.info(f"\nTotal execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    test_parallel_training()
