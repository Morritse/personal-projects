# Custom Technical Indicator System

A trading system built around custom technical indicators with a focus on VWAP (Volume Weighted Average Price) and OBV (On-Balance Volume) analysis.

## Overview

This project implements a trading system that combines custom technical indicators with traditional analysis methods. It features specialized implementations of VWAP and OBV indicators, along with comprehensive backtesting capabilities.

## Components

### Core Files
- `vwap_obv_strategy.py`: Main strategy implementation
- `backtest.py`: Backtesting engine
- `bt_strategy.py`: Backtrader strategy implementation
- `utils.py`: Utility functions
- `plot_signals.py`: Signal visualization

### Configuration
- `config.py`: Main configuration
- `config_bt.py`: Backtrader-specific config
- `config.json`: JSON configuration

### Data Management
- `download_data.py`: Data acquisition
- `fetch_and_stream.py`: Real-time data
- `fetch_coinapi_data.py`: CoinAPI integration
- `run_data.py`: Data processing

## Features

### Custom Indicators
- VWAP implementation
- OBV analysis
- Custom signal generation
- Indicator combinations
- Signal filtering

### Trading Capabilities
- Live trading execution
- Real-time analysis
- Position management
- Risk control
- Performance tracking

### Data Handling
- Multiple data sources
- Real-time streaming
- Historical data
- Data caching
- API integration

## Usage

1. Configure settings:
```python
# Edit config.py or config.json
```

2. Download historical data:
```bash
python download_data.py
```

3. Run backtesting:
```bash
python backtest.py
```

4. Execute live trading:
```bash
python live.py
```

## Requirements

- Python 3.8+
- Technical analysis libraries
- Data API access
- Trading platform API
- Visualization packages

## Strategy Components

### VWAP Analysis
- Volume weighting
- Price averaging
- Trend identification
- Support/resistance
- Entry/exit signals

### OBV Implementation
- Volume flow
- Price confirmation
- Trend validation
- Divergence detection
- Signal generation

### Combined Analysis
- Indicator confluence
- Signal validation
- Trend confirmation
- Volume verification
- Pattern recognition

## Data Sources

- Market price data
- Volume information
- CoinAPI integration
- Real-time feeds
- Historical databases

## Backtesting

- Historical performance analysis
- Strategy optimization
- Parameter tuning
- Risk assessment
- Performance metrics

## Future Enhancements

1. Additional custom indicators
2. Enhanced backtesting capabilities
3. Machine learning integration
4. Real-time optimization
5. Extended market coverage

## Directory Structure

```
my_indicator/
├── archive/           # Archived components
├── cache_data/        # Cached market data
├── backtest.py        # Backtesting engine
├── config.py         # Main configuration
├── live.py           # Live trading
└── utils.py          # Utility functions
