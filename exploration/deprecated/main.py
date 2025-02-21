import asyncio
from deprecated.strategy import TradingStrategy

async def main():
    strategy = TradingStrategy()
    await strategy.run()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
