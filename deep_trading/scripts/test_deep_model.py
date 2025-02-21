#!/usr/bin/env python3
import logging
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import traceback
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from momentum_ai_trading.models.deep_model import DeepTradingModel
from momentum_ai_trading.utils.deep_data_utils import prepare_features_for_prediction, clean_features, validate_features
from momentum_ai_trading.models.deep_backtest import DeepModelBacktester
from momentum_ai_trading.pipelines.data_pipeline import fetch_daily_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_data(df, symbol):
    """Validate data for numerical issues."""
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
        mean = df[col].mean()
        std = df[col].std()
        if (df[col].abs() > mean + 5*std).any():
            issues.append(f"Extreme values found in {col}")
    
    if issues:
        logger.warning(f"\nData validation issues for {symbol}:")
        for issue in issues:
            logger.warning(f"- {issue}")
        return False
    
    return True

def prepare_data_for_symbol(symbol):
    """Prepare and validate data for a single symbol."""
    try:
        # Get data for the last 2 years
        end_date = datetime.now()
        start_date = end_date.replace(year=end_date.year - 2)
        
        logger.info(f"Downloading and preparing data...")
        # Get raw data using existing pipeline
        raw_data = fetch_daily_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        logger.info(f"Raw data shape: {raw_data.shape}")
        
        if raw_data.empty:
            logger.error(f"No data received for {symbol}")
            return None
        
        # Clean and validate data
        cleaned_data = clean_features(raw_data)
        if not validate_features(cleaned_data, symbol):
            logger.warning(f"Data validation failed for {symbol}")
            return None
        
        logger.info(f"Data processed successfully")
        logger.info(f"Available columns: {cleaned_data.columns.tolist()}")
        return cleaned_data
    
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def test_multiple_instruments():
    """Test deep learning trading system with stability monitoring."""
    logger.info("\nTesting Deep Learning Trading System")
    logger.info("===================================")
    
    # Define instruments
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'META']
    market_etfs = ['SPY', 'QQQ', 'UVXY', 'SQQQ', 'TLT']
    instruments = tech_stocks + market_etfs
    
    logger.info(f"Instruments: {', '.join(instruments)}")
    logger.info("\nTech Stocks:")
    for stock in tech_stocks:
        logger.info(f"- {stock}")
    logger.info("\nMarket Context ETFs:")
    for etf in market_etfs:
        logger.info(f"- {etf}")
    
    logger.info("\nThis will:")
    logger.info("1. Download and prepare data for all instruments")
    logger.info("2. Train model on combined data")
    logger.info("3. Run backtest on tech stocks")
    
    try:
        # Create data directory if it doesn't exist
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("data/deep_models").mkdir(parents=True, exist_ok=True)
        Path("results/deep_backtest").mkdir(parents=True, exist_ok=True)
        
        # Process each instrument
        all_data = []
        for symbol in instruments:
            logger.info(f"\nProcessing {symbol}...")
            
            # Prepare and validate data
            symbol_data = prepare_data_for_symbol(symbol)
            if symbol_data is not None:
                logger.info(f"Data shape for {symbol}: {symbol_data.shape}")
                all_data.append(symbol_data)
                logger.info(f"Successfully processed {symbol}")
            else:
                logger.error(f"Skipping {symbol} due to data issues")
        
        if not all_data:
            raise ValueError("No valid data available for training")
        
        # Combine all data
        combined_data = pd.concat(all_data, axis=0)
        logger.info(f"\nCombined data shape: {combined_data.shape}")
        logger.info(f"Combined data columns: {combined_data.columns.tolist()}")
        
        # Create sequences for training
        logger.info("\nPreparing sequences for training...")
        train_dataset, val_dataset, n_features = prepare_features_for_prediction(
            combined_data, seq_length=60
        )
        logger.info(f"Number of features: {n_features}")
        
        # Initialize and train model
        logger.info("\nTraining model...")
        model = DeepTradingModel(seq_length=60, n_features=n_features)
        
        # Train with monitoring
        history = model.fit(train_dataset, val_dataset, epochs=10)  # Reduced epochs for testing
        
        # Save model and scaler
        model_path = Path("data/deep_models/deep_trading_model_a100.keras")
        model.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Run backtests
        logger.info("\nRunning backtests...")
        for symbol in tech_stocks:
            logger.info(f"\nBacktesting {symbol}...")
            try:
                # Get symbol data
                symbol_data = prepare_data_for_symbol(symbol)
                if symbol_data is None:
                    continue
                
                logger.info(f"Backtest data shape: {symbol_data.shape}")
                logger.info(f"Backtest data columns: {symbol_data.columns.tolist()}")
                
                # Create backtester
                backtester = DeepModelBacktester(
                    model=model.model,
                    symbol=symbol,
                    risk_free_rate=0.02
                )
                
                # Run backtest
                trades, metrics = backtester.backtest(symbol_data)
                
                # Save results
                trades_file = Path(f"results/deep_backtest/trades_{symbol}_{datetime.now().strftime('%Y%m%d')}.csv")
                metrics_file = Path(f"results/deep_backtest/metrics_{symbol}_{datetime.now().strftime('%Y%m%d')}.txt")
                
                trades.to_csv(trades_file)
                with open(metrics_file, 'w') as f:
                    for metric, value in metrics.items():
                        f.write(f"{metric}: {value}\n")
                
                logger.info(f"Backtest results saved for {symbol}")
                logger.info(f"Metrics: {metrics}")
                
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
    
    except Exception as e:
        logger.error(f"\nError occurred: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_multiple_instruments()
