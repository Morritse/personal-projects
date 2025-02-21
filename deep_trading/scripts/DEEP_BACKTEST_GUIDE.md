# Deep Learning Trading System Guide

## Overview
The deep learning trading system consists of:
1. A CNN-LSTM hybrid model with attention mechanism
2. Advanced feature engineering
3. Comprehensive backtesting framework

## Running the System

### 1. Basic Usage
```python
python scripts/run_deep_backtest.py
```
This will:
1. Check for an existing trained model
2. Train a new model if none exists
3. Run backtests on AAPL, MSFT, and GOOGL
4. Save results and visualizations

### 2. Directory Structure
```
data/
├── deep_models/              # Model files
│   ├── deep_trading_model_a100.keras
│   ├── best_model.keras
│   └── training_history.csv
├── processed/               # Processed data
└── raw/                    # Raw price data

results/
└── deep_backtest/          # Backtest results
    ├── trades_AAPL_20241129.csv
    ├── metrics_AAPL_20241129.csv
    └── performance_AAPL_20241129.png
```

## Expected Outputs

### 1. Model Training
```
Starting Deep Learning Model Training
===================================
Processing AAPL...
Processing MSFT...
Processing GOOGL...

Preparing training data...
Number of features: 17

Training model...
Epoch 1/100
[====================] - loss: 0.0123 - val_loss: 0.0098
...
```

### 2. Backtest Results
For each symbol, you'll see:
```
Backtest Results for AAPL:
========================================
Initial Capital: $100,000.00
Final Value: $105,708.83
Total Return: 5.71%
Annual Return: 2.84%
Sharpe Ratio: 0.58
Max Drawdown: 3.39%
Total Trades: 245
Win Rate: 54.29%
```

### 3. Saved Files

1. Model Files:
   - `deep_trading_model_a100.keras`: Final trained model
   - `best_model.keras`: Best model checkpoint
   - `training_history.csv`: Training metrics history

2. Backtest Results (per symbol):
   - `trades_{symbol}_{date}.csv`: Detailed trade history
   - `metrics_{symbol}_{date}.csv`: Performance metrics
   - `performance_{symbol}_{date}.png`: Performance visualization

## Performance Metrics Explained

1. Return Metrics:
   - Total Return: Overall portfolio return
   - Annual Return: Annualized return rate
   - Daily Returns: Day-to-day percentage changes

2. Risk Metrics:
   - Sharpe Ratio: Risk-adjusted return (higher is better)
   - Max Drawdown: Largest peak-to-trough decline
   - Annual Volatility: Standard deviation of returns

3. Trading Metrics:
   - Total Trades: Number of trades executed
   - Win Rate: Percentage of profitable trades
   - Average Trade Return: Mean return per trade

## Customization Options

1. Model Parameters:
```python
history = train_deep_model(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    seq_length=60,  # Sequence length for predictions
    batch_size=256, # Training batch size
    epochs=100      # Training epochs
)
```

2. Backtest Parameters:
```python
results = run_deep_backtest(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    initial_capital=100000.0
)
```

## Performance Visualization

The system generates performance plots showing:
1. Portfolio value over time
2. Daily returns distribution
3. Drawdown analysis
4. Trade entry/exit points

## Best Practices

1. Data Quality:
   - Use sufficient historical data (5+ years recommended)
   - Ensure data is clean and properly processed
   - Include multiple market regimes

2. Model Training:
   - Monitor training/validation metrics
   - Use early stopping to prevent overfitting
   - Save best model checkpoint

3. Backtesting:
   - Account for transaction costs
   - Use realistic position sizing
   - Consider market impact

4. Performance Analysis:
   - Compare against benchmarks
   - Analyze different market conditions
   - Consider risk-adjusted metrics

## Troubleshooting

1. Memory Issues:
   - Reduce batch size
   - Decrease sequence length
   - Process fewer symbols simultaneously

2. Training Issues:
   - Check learning rate
   - Monitor loss curves
   - Verify data preprocessing

3. Backtest Issues:
   - Verify data availability
   - Check position calculations
   - Validate trade execution logic

## Next Steps

1. Model Improvements:
   - Add more technical indicators
   - Implement ensemble methods
   - Fine-tune hyperparameters

2. Backtesting Enhancements:
   - Add position sizing rules
   - Implement stop-loss orders
   - Add portfolio constraints

3. Production Deployment:
   - Add real-time data feeds
   - Implement risk management
   - Add monitoring and alerts
