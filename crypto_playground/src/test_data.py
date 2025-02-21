import asyncio
import json
import logging
import numpy as np
import talib
from datetime import datetime, timezone, timedelta
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_data():
    """Test data fetching and indicator calculation with real data."""
    try:
        # Load credentials
        load_dotenv()
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Initialize client
        data_client = CryptoHistoricalDataClient(api_key, secret_key)
        
        # Fetch some historical data
        logger.info("\nFetching historical data...")
        request = CryptoBarsRequest(
            symbol_or_symbols="BTC/USD",
            timeframe=TimeFrame(5, TimeFrame.Minute),
            start=datetime.now(timezone.utc) - timedelta(days=1),
            end=datetime.now(timezone.utc)
        )
        
        bars = data_client.get_crypto_bars(request)
        
        # Convert to numpy arrays
        logger.info("\nConverting data to numpy arrays...")
        data = {
            'timestamp': [],
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        }
        
        for bar in bars["BTC/USD"]:
            data['timestamp'].append(bar.timestamp)
            data['open'].append(bar.open)
            data['high'].append(bar.high)
            data['low'].append(bar.low)
            data['close'].append(bar.close)
            data['volume'].append(bar.volume)
            
        # Convert lists to numpy arrays
        for key in ['open', 'high', 'low', 'close', 'volume']:
            data[key] = np.array(data[key], dtype=float)
            
        # Print data info
        logger.info(f"\nData shape: {len(data['close'])} bars")
        logger.info(f"Time range: {data['timestamp'][0]} to {data['timestamp'][-1]}")
        logger.info(f"\nSample of close prices: {data['close'][-5:]}")
        
        # Test indicators with real data
        logger.info("\nTesting indicators with real data...")
        
        # Test ADX
        logger.info("\nCalculating ADX:")
        adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        logger.info(f"ADX shape: {adx.shape}")
        logger.info(f"Last ADX value: {adx[-1]}")
        
        # Test MACD
        logger.info("\nCalculating MACD:")
        macd, macdsignal, macdhist = talib.MACD(data['close'])
        logger.info(f"MACD shape: {macd.shape}")
        logger.info(f"Last MACD value: {macd[-1]}")
        
        # Test RSI
        logger.info("\nCalculating RSI:")
        rsi = talib.RSI(data['close'])
        logger.info(f"RSI shape: {rsi.shape}")
        logger.info(f"Last RSI value: {rsi[-1]}")
        
        logger.info("\nAll data tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in data test: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_data())
