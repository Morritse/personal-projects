import asyncio
import os
import logging
import signal
from dotenv import load_dotenv
from trader import CryptoTrader
from logging_config import setup_logging

# Setup logging
logger = setup_logging()

async def main():
    """Main entry point for the trading system."""
    # Print startup banner
    print("=" * 60)
    print("CRYPTO TRADING SYSTEM - OPTIMIZED FOR HIGH VOLATILITY")
    print("=" * 60)
    print("\nSignal Weights:")
    print("- Trend Analysis:    40% (ADX DMI MACD)")
    print("- Momentum Analysis: 30% (RSI Stochastic CCI)")
    print("- Volatility Analysis: 30% (BB ATR Squeeze)")
    print("\nTrading Parameters:")
    print("- Base Position Size: 2% of capital")
    print("- Max Position Size: 7% of capital")
    print("- Stop Loss: 1.37x ATR")
    print("- Take Profit: 1.21x ATR")
    print("- Min Confidence: 70%")
    print("\nSpecial Features:")
    print("- Squeeze Detection & Signal Boost")
    print("- Trend Confirmation Required")
    print("- Dynamic Position Sizing")
    print("- Real-time Risk Management")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    # Load environment variables
    load_dotenv()
    
    # Verify API keys
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        logger.error("Missing API keys. Please check your .env file.")
        return
        
    # Initialize trader
    trader = CryptoTrader(
        symbols=["BTC/USD", "ETH/USD"],
        bar_timeframe=5,          # 5-minute bars
        lookback_periods=100,     # For technical analysis
        update_interval=1,        # Check signals every minute
        trade_interval=5          # Trade decisions every 5 minutes
    )
    
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s, loop, trader)))
        
    try:
        # Start trading
        await trader.run()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        await trader.stop()
        raise
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        loop.remove_signal_handler(signal.SIGINT)

async def shutdown(sig, loop, trader):
    """Cleanup tasks tied to the service's shutdown."""
    logger.info(f"Received exit signal {sig.name}...")
    
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    for task in tasks:
        task.cancel()
        
    logger.info("Stopping trader...")
    await trader.stop()
    
    logger.info(f"Waiting for {len(tasks)} tasks to complete...")
    await asyncio.gather(*tasks, return_exceptions=True)
    
    loop.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Clean exit handled by signal handler
