from dotenv import load_dotenv
import os

load_dotenv()

# Alpaca configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_PAPER = True  # Set to False for live trading
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

##############################################################################
# Symbol Universe
##############################################################################
SYMBOLS = [
    # Mega-cap Tech (High liquidity)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
    
    # Major ETFs (High volume)
    'SPY', 'QQQ', 'IWM',
    
    # High-Movement Tech
    'AMD', 'COIN', 'PLTR', 'NET', 'CRWD', 'DDOG',
    
    # Popular Trading Stocks
    'NIO', 'RIVN', 'LCID', 'SOFI', 'RBLX'
]

##############################################################################
# Timeframes & Bars
##############################################################################
# Switch from 1Min to 5Min bars to reduce noise and whipsaws
TIMEFRAME = '5Min'

# Logging verbosity
VERBOSE_INDICATORS = False   # Detailed indicator calculations
VERBOSE_DATA = True          # Detailed data processing

# Historical data to fetch
HISTORICAL_BARS = 200  # For 5-min data, ~200 bars covers ~16.5 hours of market time
UPDATE_INTERVAL = 60   # Seconds between updates

##############################################################################
# Indicator Settings
##############################################################################
TIMEFRAME_INDICATORS = {
    'primary': {  # 5-min for entries/exits
        'EMA_9': 9,       # Fast trend
        'EMA_20': 20,     # Slow trend
        'VWAP': None,     # Price relative to volume-weighted average
        'RSI': 14,        # More standard RSI period
        'DONCHIAN': 20,   # 20-bar channel (now 20 *5min bars*)
        'ATR': 14         # 14-bar ATR (also on 5min bars)
    },
    'trend': {    # 15-min or 30-min for higher-level trend confirmation
        'EMA_9': 9,
        'EMA_20': 20,
        'VWAP': None
    }
}

# Timeframes in minutes
TIMEFRAMES = {
    'primary': 5,    # 5-minute bars for entries/exits
    'trend': 15      # 15-minute for higher-level trend
}

##############################################################################
# Risk Management
##############################################################################
POSITION_RISK_PERCENT = 0.01   # Risk 1% of account equity per trade
MAX_DAILY_LOSS = 0.02          # Stop trading if down 2% on the day
MAX_TRADES_PER_DAY = 4         # Limit to 4 trades per day

##############################################################################
# Stock Selection Criteria
##############################################################################
VOLUME_THRESHOLD = 2_000_000        # Min 2M shares/day to ensure solid liquidity
VOLATILITY_THRESHOLD = 0.015        # Min 1.5% daily price move for "active" stocks
RELATIVE_VOLUME_THRESHOLD = 2.0     # 2x average volume indicates unusual activity

##############################################################################
# Signal Categories
##############################################################################
SIGNAL_TYPES = {
    'BREAKOUT': {
        'donchian_break': True,   # Price breaks Donchian channel
        'volume_confirm': True,   # Volume > 2x average
        'above_vwap': True,       # Price > VWAP
        'rsi_minimum': 50         # RSI > 50 for bullish momentum
    },
    'PULLBACK': {
        'ema_support': True,      # Price bounces off 9 EMA
        'volume_confirm': True,   # Volume spike on bounce
        'above_vwap': True        # Maintains VWAP as support
    }
}

##############################################################################
# Stop Loss Parameters
##############################################################################
STOP_TYPES = {
    # Increase fixed stop to 1.5 x ATR (was 1.0) to account for 5-min bar volatility
    'FIXED': 1.5,              
    'TRAILING': {
        'indicator': 'EMA_9',
        # Increase buffer from 0.001 (0.1%) to 0.002 (0.2%) to avoid frequent whipsaws
        'buffer': 0.002
    }
}
