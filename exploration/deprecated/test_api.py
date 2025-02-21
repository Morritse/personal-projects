import os
import json
import time
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('ALPACA_API_KEY')
API_SECRET = os.getenv('ALPACA_SECRET_KEY')
PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

HEADERS = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Content-Type": "application/json"
}

def print_json(data):
    """Pretty print JSON data"""
    print(json.dumps(data, indent=2))

def test_endpoints():
    """Test various Alpaca API endpoints"""
    
    print("\nTesting Alpaca API Connectivity...")
    print("================================")

    # 1. Test Account Information
    print("\n1. Testing Account Information (Paper Trading)")
    print("-------------------------------------------")
    response = requests.get(f"{PAPER_URL}/v2/account", headers=HEADERS)
    print_json(response.json())

    # 2. Test Market Data
    print("\n2. Testing Market Data API (Latest SPY bar)")
    print("-------------------------------------------")
    # Format dates in RFC-3339 format as required by Alpaca v2 API
    end = datetime.now()
    start = end - timedelta(minutes=5)
    params = {
        "timeframe": "1Min",
        "limit": 1,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    response = requests.get(
        f"{DATA_URL}/v2/stocks/SPY/bars",
        headers=HEADERS,
        params=params
    )
    print_json(response.json())

    # 3. Test Asset Information
    print("\n3. Testing Asset Information (SPY)")
    print("-------------------------------------------")
    response = requests.get(f"{PAPER_URL}/v2/assets/SPY", headers=HEADERS)
    print_json(response.json())

    # 4. Test Clock
    print("\n4. Testing Clock (Market Hours)")
    print("-------------------------------------------")
    response = requests.get(f"{PAPER_URL}/v2/clock", headers=HEADERS)
    print_json(response.json())

    # 5. Test Positions
    print("\n5. Testing Position (if any exists for SPY)")
    print("-------------------------------------------")
    try:
        response = requests.get(f"{PAPER_URL}/v2/positions/SPY", headers=HEADERS)
        print_json(response.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print("No position found for SPY")
        else:
            raise

    # Check market hours before placing order
    clock_response = requests.get(f"{PAPER_URL}/v2/clock", headers=HEADERS)
    clock_data = clock_response.json()
    is_market_open = clock_data.get('is_open', False)

    # 6. Place Test Order
    print("\n6. Place Test Market Order (Paper Trading)")
    print("-------------------------------------------")
    if not is_market_open:
        print("Market is currently closed. Order will be queued for next market open.")
        
    order_data = {
        "symbol": "SPY",
        "qty": "1",
        "side": "buy",
        "type": "market",
        "time_in_force": "day"
    }
    response = requests.post(
        f"{PAPER_URL}/v2/orders",
        headers=HEADERS,
        json=order_data
    )
    print_json(response.json())

    # Wait for order to process
    time.sleep(2)

    # 7. List Orders
    print("\n7. List Open Orders")
    print("-------------------------------------------")
    response = requests.get(f"{PAPER_URL}/v2/orders", headers=HEADERS)
    print_json(response.json())

    # 8. Cancel All Orders
    print("\n8. Cancel All Orders")
    print("-------------------------------------------")
    response = requests.delete(f"{PAPER_URL}/v2/orders", headers=HEADERS)
    print("All orders cancelled" if response.status_code == 207 else "No orders to cancel")

    print("\nAPI Tests Complete")
    print("==================")

if __name__ == "__main__":
    try:
        test_endpoints()
    except requests.exceptions.RequestException as e:
        print(f"\nError testing API: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
