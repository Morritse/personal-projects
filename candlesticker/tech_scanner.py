"""
Technical analysis scanner for any list of symbols
Usage: python tech_scanner.py AAPL MSFT GOOGL
       python tech_scanner.py --file symbols.txt
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

# Initialize Alpaca API
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"

def get_technical_data(symbol: str) -> dict:
    """Get technical analysis data for a symbol"""
    try:
        # Get historical data
        url = f"{ALPACA_PAPER_URL}/v2/stocks/{symbol}/bars"
        headers = {
            'APCA-API-KEY-ID': ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': ALPACA_SECRET_KEY
        }
        params = {
            'timeframe': '1D',
            'limit': 200,  # Get enough data for 200-day SMA
            'adjustment': 'raw'
        }
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get('bars'):
            return None
            
        # Convert to pandas DataFrame
        df = pd.DataFrame(data['bars'])
        df['date'] = pd.to_datetime(df['t'])
        df.set_index('date', inplace=True)
        
        # Calculate technical indicators
        # Moving averages
        df['sma_20'] = df['c'].rolling(window=20).mean()
        df['sma_50'] = df['c'].rolling(window=50).mean()
        df['sma_200'] = df['c'].rolling(window=200).mean()
        
        # RSI
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['c'].ewm(span=12, adjust=False).mean()
        exp2 = df['c'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['c'].rolling(window=20).mean()
        bb_std = df['c'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ATR
        high_low = df['h'] - df['l']
        high_close = np.abs(df['h'] - df['c'].shift())
        low_close = np.abs(df['l'] - df['c'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # Get latest row
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Determine trend
        trend = "Uptrend" if all(latest[f'sma_{p}'] < latest['c'] for p in [20, 50, 200]) else \
                "Downtrend" if all(latest[f'sma_{p}'] > latest['c'] for p in [20, 50, 200]) else \
                "Mixed"
                
        # Determine support/resistance
        pivots = {
            'P': latest['c'],
            'R1': latest['c'] + (latest['h'] - latest['l']),
            'S1': latest['c'] - (latest['h'] - latest['l']),
            'R2': latest['c'] + 2 * (latest['h'] - latest['l']),
            'S2': latest['c'] - 2 * (latest['h'] - latest['l'])
        }
        
        return {
            'symbol': symbol,
            'price': round(latest['c'], 2),
            'change': round((latest['c'] - prev['c']) / prev['c'] * 100, 2),
            'volume': int(latest['v']),
            'trend': trend,
            'technicals': {
                'sma_20': round(latest['sma_20'], 2),
                'sma_50': round(latest['sma_50'], 2),
                'sma_200': round(latest['sma_200'], 2),
                'rsi': round(latest['rsi'], 2),
                'macd': {
                    'macd': round(latest['macd'], 3),
                    'signal': round(latest['signal'], 3),
                    'hist': round(latest['macd_hist'], 3)
                },
                'bollinger_bands': {
                    'upper': round(latest['bb_upper'], 2),
                    'middle': round(latest['bb_middle'], 2),
                    'lower': round(latest['bb_lower'], 2)
                },
                'atr': round(latest['atr'], 2)
            },
            'support_resistance': {
                name: round(level, 2) 
                for name, level in pivots.items()
            }
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None

def analyze_symbols(symbols):
    """Analyze a list of symbols"""
    results = {}
    
    print(f"\nAnalyzing {len(symbols)} symbols...")
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        data = get_technical_data(symbol)
        if data:
            results[symbol] = data
            
            # Print summary
            print(f"Price: ${data['price']} ({data['change']}%)")
            print(f"Trend: {data['trend']}")
            print(f"RSI: {data['technicals']['rsi']}")
            print(f"MACD Histogram: {data['technicals']['macd']['hist']}")
            
            # Print signals
            signals = []
            tech = data['technicals']
            
            # Trend signals
            price = data['price']
            if price > tech['sma_20'] > tech['sma_50'] > tech['sma_200']:
                signals.append("Strong uptrend (price > 20MA > 50MA > 200MA)")
            elif price < tech['sma_20'] < tech['sma_50'] < tech['sma_200']:
                signals.append("Strong downtrend (price < 20MA < 50MA < 200MA)")
                
            # RSI signals
            if tech['rsi'] > 70:
                signals.append("Overbought (RSI > 70)")
            elif tech['rsi'] < 30:
                signals.append("Oversold (RSI < 30)")
                
            # MACD signals
            if tech['macd']['hist'] > 0 and abs(tech['macd']['hist']) > abs(tech['macd']['hist'] * 1.1):
                signals.append("MACD histogram increasing (bullish)")
            elif tech['macd']['hist'] < 0 and abs(tech['macd']['hist']) > abs(tech['macd']['hist'] * 1.1):
                signals.append("MACD histogram decreasing (bearish)")
                
            # Bollinger Band signals
            bb = tech['bollinger_bands']
            if price > bb['upper']:
                signals.append("Price above upper BB (overbought)")
            elif price < bb['lower']:
                signals.append("Price below lower BB (oversold)")
                
            if signals:
                print("\nSignals:")
                for signal in signals:
                    print(f"- {signal}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'technical_analysis_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis saved to {output_file}")
    return results

def main():
    parser = argparse.ArgumentParser(description='Technical analysis scanner')
    parser.add_argument('--file', help='File containing symbols (one per line)')
    parser.add_argument('symbols', nargs='*', help='List of symbols to analyze')
    
    args = parser.parse_args()
    
    if args.file:
        with open(args.file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif args.symbols:
        symbols = args.symbols
    else:
        parser.print_help()
        sys.exit(1)
    
    analyze_symbols(symbols)

if __name__ == '__main__':
    main()
