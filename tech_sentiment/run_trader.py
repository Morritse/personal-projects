#!/usr/bin/env python3
import time
from datetime import datetime
from trader import Trader

def run_trading_loop():
    """
    Run trader and monitor positions:
    1. Place initial orders
    2. Monitor every 5 minutes
    3. Exit positions on signals
    4. Place new orders as needed
    """
    trader = Trader()
    interval = 300  # 5 minutes
    
    while True:
        try:
            print(f"\n=== Trading Loop Starting at {datetime.now()} ===")
            
            # Run full trading cycle
            trader.analyze_and_trade()
            
            print(f"\nSleeping for {interval} seconds...")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nTrading loop stopped by user")
            break
        except Exception as e:
            print(f"\nError in trading loop: {str(e)}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    print("\nStarting trading system...")
    run_trading_loop()
