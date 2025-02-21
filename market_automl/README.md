# Market AutoML Trading Project

This project uses Google Cloud AutoML to predict stock movements using technical indicators and market correlations. It focuses on 5 major tech stocks and their relationships with major market indices.

## Project Structure

```
market_automl/
├── config.py              # Configuration settings and parameters
├── data_collection.py     # Alpaca API data collection
├── feature_engineering.py # Technical indicator generation using pandas-ta
├── automl_handler.py      # Google Cloud AutoML integration
├── main.py               # Main execution script
└── README.md             # Project documentation
```

## Features

- Data collection using Alpaca API
- Technical indicators using pandas-ta library
- Cross-correlation analysis between tech stocks and indices
- Automated feature engineering and selection
- Google Cloud AutoML integration for model training
- Visualization of results and model performance

## Setup

1. Install required dependencies:
```bash
pip install google-cloud-automl google-cloud-storage pandas-ta alpaca-trade-api python-dotenv pandas numpy matplotlib
```

2. Set up environment variables:
Create a `.env` file with your API keys:
```
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
```

3. Set up Google Cloud credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

## Usage

1. Run the complete pipeline:
```bash
python main.py
```

This will:
- Collect historical data for configured stocks
- Generate technical indicators and features
- Train AutoML models for each stock
- Create visualizations and evaluation metrics

2. Results will be saved in:
- `data/raw/`: Raw stock data
- `data/processed/`: Processed features
- `results/`: Model evaluations and visualizations

## Configuration

Edit `config.py` to modify:
- Stock symbols and indices to track
- Technical indicator parameters
- Feature engineering settings
- AutoML training parameters

## Technical Indicators

The project uses the following technical indicators from pandas-ta:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Simple Moving Averages (20, 50, 200 days)
- Exponential Moving Average (20 days)
- ATR (Average True Range)

## Feature Groups

1. Price Features:
- OHLCV data
- Lagged price and volume data

2. Technical Indicators:
- Momentum indicators (RSI, MACD)
- Volatility indicators (Bollinger Bands, ATR)
- Trend indicators (SMAs, EMA)

3. Market Features:
- Cross-correlations with index funds
- Aggregate market trends
- Market volatility metrics

## Model Training

The project uses Google Cloud AutoML Tables to train regression models for each stock. The models predict future returns based on the engineered features.

## Results Analysis

The project generates:
- Feature correlation heatmaps
- Feature importance plots
- Model evaluation metrics (RMSE, MAE, R²)
- Performance visualizations

## Notes

- Default configuration uses daily data with a 2-year lookback period
- Models are trained to predict 5-day forward returns
- Feature engineering includes time-lagged features up to 21 days
