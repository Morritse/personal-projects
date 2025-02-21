# Trading pairs configuration
TECH_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']
INDEX_FUNDS = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']

# Feature engineering parameters
TIME_LAGS = [1, 2, 3, 5, 10, 21]  # Days to look back for lagged features
CORRELATION_WINDOW = 30  # Window for rolling correlation calculations

# Technical indicators configuration
FAST_PERIOD = 12
SLOW_PERIOD = 26
SIGNAL_PERIOD = 9
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2

# Alpaca API configuration
import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_BASE_URL = 'https://paper-api.alpaca.markets'  # Paper trading URL

# Google Cloud configuration
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
GOOGLE_CLOUD_REGION = 'us-central1'  # Default region
MODEL_DISPLAY_NAME = 'trading_model_v1'
DATASET_DISPLAY_NAME = 'trading_dataset_v1'

# Data parameters
TIMEFRAME = '1Day'
LOOKBACK_PERIOD = '2 years'  # Historical data to fetch

# Feature groups
PRICE_FEATURES = ['open', 'high', 'low', 'close', 'volume']
MOMENTUM_INDICATORS = ['rsi', 'macd', 'macd_signal', 'macd_hist']
VOLATILITY_INDICATORS = ['bb_upper', 'bb_middle', 'bb_lower', 'atr']
TREND_INDICATORS = ['sma_20', 'sma_50', 'sma_200', 'ema_20']
