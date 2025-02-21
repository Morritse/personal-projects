import logging
import signal
import sys
from trader import CryptoTrader
import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal. Exiting...")
    sys.exit(0)

def main():
    """Main entry point for the crypto trading bot"""
    try:
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Starting Crypto Trading Bot...")
        logger.info(f"Trading the following symbols: {config.SYMBOLS}")
        logger.info(f"Timeframe: {config.TIMEFRAME_AMOUNT}{config.TIMEFRAME_UNIT}")
        
        # Initialize and run the trader
        trader = CryptoTrader()
        trader.run_trading_strategy()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
