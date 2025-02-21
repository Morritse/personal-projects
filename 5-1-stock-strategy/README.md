# 5-1 Stock Strategy

A unified stock trading strategy implementation that combines multiple technical indicators with a 5-1 time period analysis approach.

## Overview

This project implements a trading strategy that analyzes stock movements across different time periods (5-minute and 1-minute intervals) to identify trading opportunities. It includes both backtesting capabilities and live trading functionality.

## Components

### Configuration
- `config.py`: Main configuration settings for the strategy
- `bear_config.py`: Specific settings for bear market conditions

### Data Management
- `download_data.py`: Historical data acquisition module
- Supports multiple data sources and timeframes

### Trading Implementation
- `unified.py`: Core strategy implementation combining multiple indicators
- `live_trade.py`: Real-time trading execution module
- `run_unified.py`: Strategy runner with unified signal generation

## Features

- Multi-timeframe analysis (5-minute and 1-minute)
- Adaptive configuration for different market conditions
- Real-time trading capabilities
- Historical data backtesting
- Unified signal generation system

## Usage

1. Configure settings:
```python
# Edit config.py or bear_config.py based on market conditions
```

2. Download historical data:
```bash
python download_data.py
```

3. Run live trading:
```bash
python live_trade.py
```

4. Run unified strategy:
```bash
python run_unified.py
```

## Requirements

- Python 3.8+
- Market data access (API keys required)
- Trading account access for live trading

## Strategy Logic

The 5-1 strategy combines:
- 5-minute chart for trend direction
- 1-minute chart for entry/exit signals
- Technical indicator confluence
- Volume analysis
- Price action patterns

## Risk Management

- Position sizing based on account equity
- Stop-loss implementation
- Profit taking rules
- Market condition adaptation
