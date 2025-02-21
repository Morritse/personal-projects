import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project Root Directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data Paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')
LIVE_DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'live')

# Model Paths
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'saved_models', 'momentum_model')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
PLOTS_PATH = os.path.join(RESULTS_PATH, 'plots')
LOGS_PATH = os.path.join(RESULTS_PATH, 'logs')

# Alpaca Trading API Configuration
BASE_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', 'YOUR_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_SECRET_KEY', 'YOUR_API_SECRET')

# Ensure directories exist
os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
os.makedirs(LIVE_DATA_PATH, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

# Trading Configuration
INITIAL_CAPITAL = 100000
COMMISSION_RATE = 0.001  # 0.1% per trade

# Validate Alpaca API credentials
if not ALPACA_API_KEY or not ALPACA_API_SECRET:
    logger.warning("WARNING: Alpaca API credentials are not set.")
    logger.warning("Please verify your API credentials.")
