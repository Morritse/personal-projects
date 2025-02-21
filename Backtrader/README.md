# Backtrader Trading System

A comprehensive trading system built on the Backtrader framework with multi-timeframe signal generation and ensemble strategy optimization.

## Overview

This project implements a sophisticated trading system that combines multiple timeframe analysis with ensemble methods for improved market prediction accuracy. It features optimized signal generation across different time horizons and advanced strategy combination techniques.

## Components

### Signal Generation
- `short_term_signals.py`: Short-term market indicators and signals
- `med_term_signals.py`: Medium-term trend analysis
- `long_term_signals.py`: Long-term market trend signals
- `vol_signals.py`: Volume-based signal generation

### Strategy Implementation
- `ensemble.py`: Base ensemble strategy implementation
- `ensemble_optimized.py`: Optimized version with parameter tuning
- `shortmed.py`: Combined short and medium term strategy

### Execution
- `run_basic.py`: Basic strategy execution
- `run_ensemble.py`: Ensemble strategy runner
- `run_optimized.py`: Optimized strategy execution

### Testing
- `test_backtrader.py`: Core system tests
- `test_short_term.py`: Short-term signal testing
- `test_shortmed.py`: Combined strategy testing

## Features

- Multi-timeframe analysis
- Ensemble strategy combination
- Signal optimization
- Volume analysis integration
- Automated parameter tuning
- Comprehensive testing suite

## Strategy Types

### 1. Short-Term
- Intraday signals
- Quick momentum indicators
- Volume price analysis
- Pattern recognition

### 2. Medium-Term
- Swing trading signals
- Trend following indicators
- Moving average systems
- Breakout detection

### 3. Long-Term
- Trend identification
- Major support/resistance
- Market regime detection
- Position trading signals

### 4. Ensemble Methods
- Strategy combination
- Weight optimization
- Performance metrics
- Risk management

## Usage

1. Run basic strategy:
```bash
python run_basic.py
```

2. Execute ensemble strategy:
```bash
python run_ensemble.py
```

3. Run optimized version:
```bash
python run_optimized.py
```

## Testing

Run the test suite:
```bash
python test_backtrader.py
python test_short_term.py
python test_shortmed.py
```

## Requirements

- Python 3.8+
- Backtrader framework
- NumPy
- Pandas
- TA-Lib (Technical Analysis Library)

## Strategy Optimization

The system includes parameter optimization for:
- Signal thresholds
- Timeframe weights
- Entry/exit rules
- Position sizing
- Risk parameters

## Risk Management

- Dynamic position sizing
- Multiple timeframe confirmation
- Ensemble-based risk assessment
- Stop-loss implementation
- Profit taking rules
