from ib_insync import *
from datetime import datetime

def reset_account():
    print("\n=== Resetting Paper Trading Account ===\n")
    
    try:
        # Connect to IB
        print("Connecting to IB...")
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=1)
        print("Connected successfully\n")
        
        # Get all positions
        positions = ib.positions()
        if not positions:
            print("No positions found - account already flat")
            return
            
        print("Current Positions:")
        for pos in positions:
            print(f"{pos.contract.symbol}: {pos.position}")
            
        print("\nClosing all positions...")
        
        # Close each position
        for pos in positions:
            contract = pos.contract
            quantity = pos.position
            
            # Ensure contract has exchange set
            if not contract.exchange:
                if contract.symbol in ['ES', 'NQ', 'RTY']:
                    contract.exchange = 'CME'
                elif contract.symbol in ['YM', 'ZB', 'ZN', 'ZF', 'ZC', 'ZW', 'ZS']:
                    contract.exchange = 'CBOT'
                elif contract.symbol in ['CL', 'NG', 'RB', 'HO']:
                    contract.exchange = 'NYMEX'
                elif contract.symbol in ['GC', 'SI', 'HG']:
                    contract.exchange = 'COMEX'
                    
            # Qualify the contract
            ib.qualifyContracts(contract)
            
            # Determine action (opposite of current position)
            action = 'SELL' if quantity > 0 else 'BUY'
            order = MarketOrder(action, abs(quantity))
            
            print(f"\nClosing {contract.symbol} on {contract.exchange}: {action} {abs(quantity)}")
            trade = ib.placeOrder(contract, order)
            
            # Wait for fill with timeout
            timeout = 10  # seconds
            start_time = datetime.now()
            while not trade.isDone():
                if (datetime.now() - start_time).seconds > timeout:
                    print(f"Order timeout for {contract.symbol}, canceling...")
                    ib.cancelOrder(order)
                    break
                ib.sleep(1)
                
            if trade.orderStatus.status == 'Filled':
                print(f"Closed at {trade.orderStatus.avgFillPrice}")
            else:
                print(f"Order not filled - Status: {trade.orderStatus.status}")
                if hasattr(trade, 'log'):
                    for entry in trade.log:
                        if entry.errorCode:
                            print(f"Error {entry.errorCode}: {entry.message}")
        
        # Verify account is flat
        final_positions = ib.positions()
        if not final_positions:
            print("\nAll positions closed successfully - account is flat")
        else:
            print("\nWarning: Some positions remain:")
            for pos in final_positions:
                print(f"{pos.contract.symbol}: {pos.position}")
                
    except Exception as e:
        print(f"\n!!! Error Resetting Account !!!")
        print(f"Error details: {str(e)}")
        raise
        
    finally:
        if 'ib' in locals() and ib.isConnected():
            ib.disconnect()
            print("\nDisconnected from IB")

if __name__ == "__main__":
    reset_account()
