import os
import json
import asyncio
import websockets
from dotenv import load_dotenv
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_crypto_stream():
    # Load credentials
    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    # Use v1beta3 crypto endpoint
    url = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
    logger.info(f"\nTesting crypto endpoint: {url}")
    logger.info(f"API Key: {api_key[:5]}...{api_key[-5:]}")
    
    try:
        async with websockets.connect(url) as websocket:
            # Should receive connection success message
            response = await websocket.recv()
            logger.info(f"Connection response: {response}")
            
            # Send authentication message
            auth_msg = {
                "action": "auth",
                "key": api_key,
                "secret": secret_key
            }
            logger.info("Sending auth message...")
            await websocket.send(json.dumps(auth_msg))
            
            # Wait for auth response
            response = await websocket.recv()
            logger.info(f"Auth response: {response}")
            
            # Subscribe to different data types
            subscribe_msg = {
                "action": "subscribe",
                "trades": ["BTC/USD", "ETH/USD"],
                "quotes": ["BTC/USD", "ETH/USD"],
                "bars": ["BTC/USD", "ETH/USD"],
                "orderbooks": ["BTC/USD", "ETH/USD"]
            }
            logger.info("Sending subscribe message...")
            await websocket.send(json.dumps(subscribe_msg))
            
            # Wait for subscription confirmation
            response = await websocket.recv()
            logger.info(f"Subscribe response: {response}")
            
            # Wait for and display different message types
            logger.info("\nWaiting for market data (10 seconds)...")
            try:
                end_time = asyncio.get_event_loop().time() + 10
                while asyncio.get_event_loop().time() < end_time:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        data = json.loads(message)
                        
                        # Pretty print based on message type
                        if isinstance(data, list):
                            for msg in data:
                                msg_type = msg.get('T')
                                if msg_type == 't':
                                    logger.info(f"Trade: {msg['S']} - Price: {msg['p']}, Size: {msg['s']}")
                                elif msg_type == 'q':
                                    logger.info(f"Quote: {msg['S']} - Bid: {msg['bp']} x {msg['bs']}, Ask: {msg['ap']} x {msg['as']}")
                                elif msg_type == 'b':
                                    logger.info(f"Bar: {msg['S']} - O: {msg['o']}, H: {msg['h']}, L: {msg['l']}, C: {msg['c']}, V: {msg['v']}")
                                elif msg_type == 'o':
                                    logger.info(f"Orderbook: {msg['S']} - {len(msg['b'])} bids, {len(msg['a'])} asks")
                    except asyncio.TimeoutError:
                        continue
                        
            except asyncio.CancelledError:
                pass
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        logger.error("\nCommon issues:")
        logger.error("1. Check if API key has crypto data permissions")
        logger.error("2. Verify account has crypto data subscription")
        logger.error("3. Ensure API key is active in Alpaca dashboard")
        logger.error("4. Check if paper trading is properly configured")

if __name__ == "__main__":
    asyncio.run(test_crypto_stream())
