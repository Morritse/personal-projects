# Long-Short Trading

Implementation of long-short equity trading strategies using statistical arbitrage.

## Structure

```
long_short/
├── main.py           # Main execution script
├── models.py         # Trading models implementation
├── utils.py          # Utility functions
├── validation.py     # Strategy validation
├── run_validation.py # Validation runner
└── test_*.py        # Test modules
```

## Components

### Trading System
- Pair trading implementation
- Market neutral strategies
- Position sizing logic
- Risk controls

### Validation
- Historical backtesting
- Performance metrics
- Risk analysis
- Strategy validation

### Execution
- Parallel processing support
- Portfolio management
- Order execution
- Position tracking

## Setup

Required packages:
```
numpy
pandas
scipy
scikit-learn
```

## Usage

Run validation:
```bash
python run_validation.py
```

Test portfolio:
```bash
python test_portfolio.py
```

## Configuration

Edit parameters in `utils.py`:
- Position sizing
- Risk limits
- Trading pairs
- Execution settings
