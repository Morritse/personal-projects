from alpaca.trading.client import TradingClient

# Alpaca credentials
API_KEY = "PKASZ48REAQARDXG66WF"
API_SECRET = "L8w2jmhDilnFSxFA9VLNMDbef0copxhf3NOTXSFH"

def cancel_all_orders():
    """Cancel all open orders"""
    print("\nCanceling all open orders...")
    
    # Initialize client
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    
    try:
        # Get all open orders
        orders = trading_client.get_orders()
        
        if not orders:
            print("No open orders found")
            return
            
        print(f"\nFound {len(orders)} open orders:")
        for order in orders:
            print(f"  {order.symbol}: {order.side} {order.qty} shares ({order.status})")
        
        # Cancel all orders
        trading_client.cancel_orders()
        print("\nAll orders canceled successfully")
        
    except Exception as e:
        print(f"Error canceling orders: {str(e)}")

if __name__ == "__main__":
    cancel_all_orders()
