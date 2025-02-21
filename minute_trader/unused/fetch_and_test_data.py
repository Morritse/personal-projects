import os
import alpaca_trade_api as tradeapi
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from vwap_obv_strategy import VAMEStrategy

# Load Alpaca credentials from .env
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_SECRET_KEY')

# Initialize Alpaca API
api = tradeapi.REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')

# Configuration for strategy
config = {
    'Risk Per Trade': 0.025,
    'MFI Period': 9,
    'VWAP Window': 50,
    'Regime Window': 20,
    'mfi_entry': 30,
    'bear_exit': 55,
    'bull_exit': 75
}

def fetch_data(symbols, start_date, end_date):
    """Fetch historical data for multiple symbols"""
    data = {}
    for symbol in symbols:
        if symbol == 'VIX':
            # Get VIX data from yfinance
            vix = yf.download('^VIX', start=start_date, end=end_date, interval='1h')
            vix = vix.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            data[symbol] = vix
        else:
            # Get other symbols from Alpaca
            bars = api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start=start_date,
                end=end_date
            ).df
            
            # Use lowercase column names directly
            bars = bars.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            data[symbol] = bars
    return data

def prepare_data(data):
    """Prepare data for strategy testing"""
    # Combine data into single DataFrame
    combined = pd.concat(data.values(), keys=data.keys(), names=['symbol'])
    
    # Calculate volatility using SPY as proxy
    if 'SPY' in data:
        spy_data = data['SPY']
        spy_data['returns'] = spy_data['close'].pct_change()
        volatility_data = spy_data
    else:
        # Use first available symbol if SPY not present
        first_symbol = list(data.keys())[0]
        volatility_data = data[first_symbol]
        volatility_data['returns'] = volatility_data['close'].pct_change()
    
    return combined, volatility_data

def test_strategy(symbol_data, vix_data):
    """Test strategy on given data"""
    strategy = VAMEStrategy(config)
    results = {}
    
    for symbol in symbol_data.index.levels[0]:
        if symbol == 'VIX':
            continue
            
        print(f"\nAnalyzing {symbol}...")
        symbol_df = symbol_data.loc[symbol]
        trades = strategy.run(symbol_df, vix_data)
        results[symbol] = trades
        
        if trades:
            print(f"\nTrade details for {symbol}:")
            for i, trade in enumerate(trades, 1):
                print(f"\nTrade {i}:")
                print(f"Action: {trade['action']}")
                print(f"Price: {trade['price']:.2f}")
                if 'size' in trade:
                    print(f"Size: {trade['size']}")
                if 'pnl' in trade:
                    print(f"PnL: ${trade['pnl']:.2f}")
                if 'reason' in trade:
                    print(f"Exit Reason: {trade['reason']}")
                if 'regime' in trade:
                    print(f"Market Regime: {trade['regime']}")
    
    return results

def main():
    # Define symbols and date range
    symbols = ['AAPL', 'MSFT', 'NVDA', 'VIX']
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    # Fetch data
    print("Fetching historical data...")
    data = fetch_data(symbols, start_date.isoformat(), end_date.isoformat())
    
    # Prepare data
    print("Preparing data for analysis...")
    symbol_data, vix_data = prepare_data(data)
    
    # Test strategy
    print("Running strategy tests...")
    results = test_strategy(symbol_data, vix_data)
    
    # Print results
    for symbol, trades in results.items():
        if symbol != 'VIX':
            print(f"\nResults for {symbol}:")
            print(f"Total trades: {len(trades)}")
            if trades:
                total_pnl = sum(t['pnl'] for t in trades if 'pnl' in t)
                print(f"Total PnL: ${total_pnl:.2f}")

if __name__ == '__main__':
    main()
