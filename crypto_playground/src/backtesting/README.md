# Backtesting and Parameter Optimization

This folder contains tools for backtesting and optimizing trading strategy parameters.

## Files

- `optimizer.py`: Grid search optimization for trading parameters
- `backtest_runner.py`: Runner script with visualization and analysis tools

## Usage

1. Run parameter optimization:
```bash
python backtest_runner.py
```

This will:
- Test different parameter combinations
- Generate performance metrics
- Create visualization plots
- Save results to the `results` directory

## Parameters Tested

The optimizer tests combinations of:

1. Signal Thresholds:
```python
'MIN_SIGNAL_STRENGTH': [0.05, 0.1, 0.15, 0.2]
'MIN_CONFIDENCE': [0.4, 0.5, 0.6, 0.7]
'MIN_COMPONENT_CONF': [0.3, 0.4, 0.5, 0.6]
```

2. Position Sizing:
```python
'BASE_POSITION_SIZE': [2500.0, 5000.0, 7500.0, 10000.0]
'MAX_POSITION_SIZE': [5000.0, 10000.0, 15000.0, 20000.0]
```

3. Risk Parameters:
```python
'STOP_LOSS_ATR': [1.0, 1.5, 2.0, 2.5]
'TAKE_PROFIT_ATR': [2.0, 2.5, 3.0, 3.5]
```

4. Signal Boosts:
```python
'TREND_BOOST': [1.2, 1.3, 1.4, 1.5]
'MOMENTUM_BOOST': [1.1, 1.2, 1.3, 1.4]
'VOLATILITY_BOOST': [1.2, 1.3, 1.4, 1.5]
'CONFIDENCE_BOOST': [1.2, 1.3, 1.4, 1.5]
```

## Output

1. JSON Results:
- Saved to `results/optimization_results_[timestamp].json`
- Contains all parameter combinations and their performance metrics

2. Visualization Plots:
- Saved to `results/optimization_plots_[timestamp].png`
- Shows:
  * Sharpe Ratio vs Profit/Loss
  * Parameter distributions for top performers
  * Win Rate vs Max Drawdown
  * Trade Frequency vs Duration

3. Console Output:
- Summary statistics
- Best performing combinations
- Parameter ranges in top 20% of results

## Metrics

The optimizer evaluates each parameter combination using:

1. Sharpe Ratio: Risk-adjusted returns
2. Profit/Loss: Total return percentage
3. Win Rate: Percentage of profitable trades
4. Max Drawdown: Largest peak-to-trough decline
5. Total Trades: Number of trades executed
6. Average Duration: Mean trade duration in minutes

## Customization

You can modify:

1. Test Range:
```python
lookback_days=30  # Amount of historical data
```

2. Number of Combinations:
```python
max_combinations=100  # Limit number of parameter combinations
```

3. Symbols:
```python
symbols=["BTC/USD", "ETH/USD"]  # Cryptocurrencies to test
```

## Example Results

The output will look like:
```
Optimization Results Summary:
--------------------------------------------------
Total Parameter Combinations Tested: 100

Best Sharpe Ratio: 2.45
Best Profit/Loss: 18.72%
Best Win Rate: 65.3%

Parameter Ranges Found in Top 20% of Results:
MIN_SIGNAL_STRENGTH:
  Range: 0.10 to 0.20
  Mean: 0.15 ± 0.03
...
```

Use these results to find optimal parameters for your trading strategy.
