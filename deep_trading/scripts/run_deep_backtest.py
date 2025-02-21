import os
import pandas as pd
from datetime import datetime, timedelta
import pytz

from momentum_ai_trading.utils.deep_data_utils import (
    prepare_deep_data, calculate_deep_indicators
)
from momentum_ai_trading.models.deep_model import DeepTradingModel
from momentum_ai_trading.models.deep_backtest import DeepBacktester
from momentum_ai_trading.pipelines.data_pipeline import prepare_daily_dataset

def run_deep_backtest(symbols=['AAPL', 'MSFT', 'GOOGL'], 
                     model_path='data/deep_models/deep_trading_model_a100.keras',
                     initial_capital=100000.0):
    """Run complete deep learning backtest pipeline."""
    
    print("\nStarting Deep Learning Trading Backtest")
    print("=======================================")
    
    # Ensure model exists
    if not os.path.exists(model_path):
        raise ValueError(f"Model not found at {model_path}. Please train the model first.")
    
    results = {}
    
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        
        # Get data
        data = prepare_daily_dataset(symbol)
        if data.empty:
            print(f"No data available for {symbol}")
            continue
            
        # Initialize backtester
        backtester = DeepBacktester(model_path, seq_length=60)
        
        # Run backtest
        print(f"Running backtest for {symbol}...")
        trades, metrics = backtester.backtest(data, initial_capital=initial_capital)
        
        # Save results
        print(f"Saving results for {symbol}...")
        backtester.save_results(trades, metrics, symbol)
        
        # Store results
        results[symbol] = {
            'trades': trades,
            'metrics': metrics
        }
        
        # Print metrics
        print(f"\nBacktest Results for {symbol}:")
        print("=" * 40)
        print(f"Initial Capital: ${metrics['Initial Capital']:,.2f}")
        print(f"Final Value: ${metrics['Final Value']:,.2f}")
        print(f"Total Return: {metrics['Total Return']:.2%}")
        print(f"Annual Return: {metrics['Annual Return']:.2%}")
        print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown: {metrics['Max Drawdown']:.2%}")
        print(f"Total Trades: {metrics['Total Trades']}")
        print(f"Win Rate: {metrics['Win Rate']:.2%}")
    
    return results

def train_deep_model(symbols=['AAPL', 'MSFT', 'GOOGL'], 
                    save_dir='data/deep_models',
                    seq_length=60,
                    batch_size=256,
                    epochs=100):
    """Train deep learning model on multiple symbols."""
    
    print("\nStarting Deep Learning Model Training")
    print("===================================")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect data from all symbols
    all_data = []
    for symbol in symbols:
        print(f"\nProcessing {symbol}...")
        data = prepare_daily_dataset(symbol)
        if not data.empty:
            processed_data = calculate_deep_indicators(data)
            all_data.append(processed_data)
    
    if not all_data:
        raise ValueError("No data available for training")
    
    # Combine data
    combined_data = pd.concat(all_data)
    combined_data.sort_index(inplace=True)
    
    # Prepare datasets
    print("\nPreparing training data...")
    train_dataset, val_dataset, n_features = prepare_deep_data(
        combined_data,
        seq_length=seq_length,
        batch_size=batch_size
    )
    
    # Initialize and compile model
    print("\nInitializing model...")
    model = DeepTradingModel(seq_length, n_features)
    model.compile_model()
    
    # Train model
    print("\nTraining model...")
    history = model.train(
        train_dataset,
        val_dataset,
        epochs=epochs,
        save_dir=save_dir
    )
    
    print("\nModel training complete!")
    return history

if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    model_path = 'data/deep_models/deep_trading_model_a100.keras'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print("Model not found. Training new model...")
        history = train_deep_model(symbols)
    
    # Run backtest
    results = run_deep_backtest(symbols, model_path)
    
    print("\nBacktesting complete!")
