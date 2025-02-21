# Deep Trading

Trading system using CNN-LSTM models with attention mechanisms for market prediction.

## Structure

```
deep_trading/
├── models/             # Model implementations
├── pipelines/          # Data and trading pipelines
├── notebooks/          # Training and analysis notebooks
├── results/           # Trading results and analysis
├── scripts/           # Execution scripts
├── tests/             # Test modules
└── utils/             # Utility functions
```

## Components

### Models
- CNN-LSTM architecture
- Attention mechanism
- Ensemble methods
- Model optimization

### Pipelines
- Data processing
- Backtesting
- Live trading
- Feature engineering

### Utils
- Data handling
- Configuration
- Model utilities
- Feature engineering

## Setup

Required packages:
```
tensorflow
numpy
pandas
scikit-learn
matplotlib
```

## Usage

Train model:
```bash
python models/train_deep_model.py
```

Run backtest:
```bash
python scripts/run_deep_backtest.py
```

## Documentation

Detailed guides available in:
- notebooks/README.md: Training guide
- scripts/DEEP_BACKTEST_GUIDE.md: Backtesting guide
