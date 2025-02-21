# Alpaca API configuration
API_KEY = "PKWEMTDE82KQ6TADJ53A"
API_SECRET = "x89znce9KMVaW2u4nXVHYbvSDlzS3DC1aNTBLBfe"
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading URL
API_DATA_URL = "https://data.alpaca.markets"   # Market data URL

# Trading parameters
SYMBOLS = [
"AAVE/USD", "BCH/USD", "BTC/USD", "DOGE/USD", "ETH/USD", "LINK/USD", "LTC/USD", "SUSHI/USD", "UNI/USD", "YFI/USD"] # Starting with major pairs

# Timeframe settings
TIMEFRAME_AMOUNT = 15  # Amount of time units
TIMEFRAME_UNIT = "minute"  # Time unit (minute, hour, day)

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
VOLUME_PERIOD = 20  # For volume moving average

# Position sizing and risk management
MAX_POSITION_SIZE = 1000  # Maximum position size in USD
STOP_LOSS_PCT = 0.02     # 2% stop loss
TAKE_PROFIT_PCT = 0.04   # 4% take profit
MAX_POSITIONS = 3        # Maximum number of concurrent positions

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
