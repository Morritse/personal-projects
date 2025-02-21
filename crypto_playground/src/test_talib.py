import numpy as np
import talib
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_talib():
    """Test TA-Lib installation and basic functionality."""
    try:
        # Print TA-Lib version
        logger.info(f"TA-Lib Version: {talib.__ta_version__}")
        
        # Create sample data
        close = np.array([
            100.0, 101.0, 102.0, 101.5, 102.5,
            103.0, 102.5, 102.0, 101.0, 100.5,
            100.0, 101.5, 102.5, 103.5, 104.0,
            103.5, 102.0, 101.5, 101.0, 102.0
        ], dtype=float)
        
        high = close + 1.0
        low = close - 1.0
        
        logger.info("\nTesting basic indicators...")
        
        # Test ADX
        logger.info("\nTesting ADX:")
        adx = talib.ADX(high, low, close, timeperiod=14)
        logger.info(f"ADX result shape: {adx.shape}")
        logger.info(f"ADX last few values: {adx[-5:]}")
        
        # Test MACD
        logger.info("\nTesting MACD:")
        macd, macdsignal, macdhist = talib.MACD(close)
        logger.info(f"MACD result shape: {macd.shape}")
        logger.info(f"MACD last few values: {macd[-5:]}")
        
        # Test RSI
        logger.info("\nTesting RSI:")
        rsi = talib.RSI(close)
        logger.info(f"RSI result shape: {rsi.shape}")
        logger.info(f"RSI last few values: {rsi[-5:]}")
        
        logger.info("\nAll TA-Lib tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error testing TA-Lib: {str(e)}")
        return False

if __name__ == "__main__":
    test_talib()
