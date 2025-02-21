# Novel Technical Analysis

A research project exploring innovative technical analysis approaches combining traditional indicators with machine learning techniques.

## Overview

This project focuses on developing and testing novel technical analysis features and indicators, with multiple iterations of machine learning models to validate their effectiveness.

## Components

### Feature Engineering
- `features.py`: Base feature implementations
- `featuresv2.py`: Enhanced feature set
- `fetch_working_data.py`: Data acquisition

### Machine Learning
- `ml.py`: Initial ML implementation
- `mlv2.py`: Second iteration
- `mlv3.py`: Third iteration
- `mlv4.py`: Latest ML version

### Data Storage
- `5_year_with_indicators/`: Processed data with indicators
- `alpaca_5yr_data/`: Raw market data
- `featuresv2_data/`: Enhanced feature data

## Features

### Technical Indicators
- Novel indicator combinations
- Custom feature engineering
- Multi-timeframe analysis
- Volume-price relationships
- Market microstructure features

### Machine Learning Models
- Multiple model iterations
- Feature importance analysis
- Performance validation
- Hyperparameter optimization
- Cross-validation frameworks

### Data Processing
- 5-year historical data
- Multiple data sources
- Feature preprocessing
- Data validation
- Quality checks

## Research Areas

### Feature Development
- Price action patterns
- Volume analysis
- Market microstructure
- Order flow indicators
- Volatility measures

### Model Evolution
- Iterative improvements
- Performance metrics
- Feature selection
- Model comparison
- Validation methods

### Data Analysis
- Historical patterns
- Market regimes
- Feature correlations
- Predictive power
- Signal stability

## Usage

1. Generate features:
```bash
python features.py  # Base features
python featuresv2.py  # Enhanced features
```

2. Run ML models:
```bash
python ml.py  # Basic model
python mlv4.py  # Latest version
```

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Technical analysis libraries

## Future Research

1. Additional feature engineering
2. Advanced ML architectures
3. Real-time analysis
4. Market regime detection
5. Automated feature selection

## Data Sources

- Alpaca API (5 years of data)
- Processed indicator data
- Feature-engineered datasets
- Market microstructure data
- Alternative data sources
