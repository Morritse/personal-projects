# Strategy Backtesting Framework

This framework provides tools for backtesting and optimizing trading strategies using historical data and machine learning techniques.

## Components

### 1. Backtester (`utils/backtester.py`)
- Simulates trading with historical data
- Tracks positions, capital, and performance metrics
- Handles position sizing and trade execution
- Calculates key metrics like Sharpe ratio, win rate, and max drawdown

### 2. Parameter Optimizer (`utils/optimizer.py`)
- Uses Bayesian optimization to find optimal strategy parameters
- Implements time series cross-validation
- Optimizes for Sharpe ratio by default
- Generates visualization plots for parameter importance and convergence

### 3. Backtest Runner (`backtest_runner.py`)
- Coordinates data fetching, backtesting, and optimization
- Manages historical data across multiple timeframes
- Saves results and generates reports
- Provides a simple interface for running experiments

## Usage

### Basic Backtesting

```python
from backtest_runner import BacktestRunner

# Initialize runner
runner = BacktestRunner(
    symbols=['SPY', 'QQQ', 'IWM', 'TLT'],
    start_date='2023-01-01',
    end_date='2023-12-31',
    timeframes=['5Min', '15Min', '1H'],
    initial_capital=100000.0
)

# Run backtest with specific parameters
metrics = runner.run_backtest({
    'trend_weight': 0.3,
    'momentum_weight': 0.3,
    'reversal_weight': 0.4,
    'breakout_threshold': 0.4,
    'strong_threshold': 0.25,
    'weak_threshold': 0.15
})
```

### Parameter Optimization

```python
# Define parameter ranges to optimize
param_ranges = {
    'trend_weight': (0.1, 0.5),
    'momentum_weight': (0.1, 0.5),
    'reversal_weight': (0.1, 0.5),
    'breakout_threshold': (0.3, 0.6),
    'strong_threshold': (0.2, 0.4),
    'weak_threshold': (0.1, 0.2)
}

# Run optimization
results = runner.optimize_parameters(
    param_ranges=param_ranges,
    n_iterations=100,  # Number of optimization iterations
    n_splits=5  # Number of cross-validation splits
)

# Run backtest with optimized parameters
metrics = runner.run_backtest(
    strategy_params=results['best_params']
)
```

## Results

The framework saves all results in the `backtesting/results` directory:

- `backtest_results_[timestamp].json`: Contains detailed backtest results including:
  * Strategy parameters
  * Performance metrics
  * Trade history
  * Equity curve

- `optimization_results_[timestamp].json`: Contains optimization results including:
  * Best parameters found
  * Cross-validation scores
  * Optimization history

- `optimization_plots_[timestamp].png`: Visualization of:
  * Parameter importance
  * Optimization convergence
  * Cross-validation performance

## Parameters to Optimize

1. Signal Weights:
- `trend_weight`: Weight for trend indicators (0.1-0.5)
- `momentum_weight`: Weight for momentum indicators (0.1-0.5)
- `reversal_weight`: Weight for reversal indicators (0.1-0.5)

2. Signal Thresholds:
- `breakout_threshold`: Threshold for breakout signals (0.3-0.6)
- `strong_threshold`: Threshold for strong signals (0.2-0.4)
- `weak_threshold`: Threshold for weak signals (0.1-0.2)

## Performance Metrics

The framework calculates several key performance metrics:

- Total Return & Return %
- Sharpe Ratio
- Win Rate
- Total Trades
- Maximum Drawdown
- Profit/Loss per Trade
- Equity Curve

## Dependencies

Required packages:
```
scikit-learn>=1.4.1
scikit-optimize>=0.9.0
matplotlib>=3.8.3
seaborn>=0.13.2
pandas>=2.2.3
numpy>=2.1.3
```

Install dependencies:
```bash
pip install -r requirements.txt
