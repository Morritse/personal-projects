"""
Generate comprehensive market data for Claude to analyze.
This script fetches both technical and fundamental data, allowing Claude to provide
nuanced analysis incorporating broader market context and intermarket relationships.

Usage:
  python market_report.py AAPL MSFT GOOGL              # Analyze specific symbols
  python market_report.py --file symbols.txt           # Read symbols from file
  python market_report.py                              # Use default symbols
"""

import json
import argparse
from datetime import datetime
from market_analyzer import fetch_market_data, MarketData

def parse_symbols():
    """Get symbols from command line or file"""
    parser = argparse.ArgumentParser(description='Market data analyzer')
    parser.add_argument('symbols', nargs='*', help='Space-separated list of symbols')
    parser.add_argument('--file', help='File containing symbols (one per line)')
    
    args = parser.parse_args()
    
    symbols = []
    if args.file:
        with open(args.file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif args.symbols:
        symbols = args.symbols
    
    if not symbols:
        # Default symbols if none provided
        symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN",  # Tech leaders
            "NVDA", "AMD",                     # Semiconductors
            "SPY", "QQQ"                       # Market benchmarks
        ]
    
    return symbols

def fetch_market_context(symbols) -> dict:
    """Fetch data for provided symbols and add relevant context"""
    market_data = {}
    
    print("\nFetching market data...")
    for symbol in symbols:
        try:
            print(f"\nProcessing {symbol}...")
            data = MarketData.from_json(fetch_market_data(symbol))
            
            # Structure data for analysis
            market_data[symbol] = {
                "price_action": {
                    "current_price": data.close,
                    "daily_change": ((data.close - data.open) / data.open) * 100,
                    "volume": data.volume
                },
                "technical": {
                    "trend": {
                        "sma_20": data.sma_20,
                        "sma_50": data.sma_50,
                        "sma_200": data.sma_200
                    },
                    "momentum": {
                        "rsi": data.rsi_14,
                        "macd": data.macd
                    },
                    "volatility": {
                        "atr": data.atr_14,
                        "bollinger_bands": data.bb_bands
                    }
                },
                "fundamental": {
                    "market_cap": data.market_cap,
                    "sector": data.sector,
                    "beta": data.beta,
                    "pe_ratio": data.pe_ratio,
                    "pb_ratio": data.pb_ratio,
                    "debt_to_equity": data.debt_to_equity
                },
                "events": {
                    "earnings_date": data.earnings_date,
                    "dividend_date": data.dividend_date
                }
            }
            
            # Print key metrics
            print(f"Price: ${data.close:.2f} ({((data.close - data.open) / data.open) * 100:.1f}%)")
            print(f"RSI: {data.rsi_14:.1f}")
            print(f"Trend: {'↑' if data.close > data.sma_20 > data.sma_50 else '↓' if data.close < data.sma_20 < data.sma_50 else '→'}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
    
    return market_data

def save_market_report(symbols=None):
    """Generate and save comprehensive market report"""
    if symbols is None:
        symbols = parse_symbols()
    
    print(f"\nAnalyzing {len(symbols)} symbols: {', '.join(symbols)}")
    timestamp = datetime.now()
    
    report = {
        "generated_at": timestamp.isoformat(),
        "analyzed_symbols": symbols,
        "market_context": {
            "analysis_timeframe": "Daily",
            "market_regime": "Analysis needed",
            "key_themes": [],
            "risk_factors": []
        },
        "market_sentiment": {
            "macro_events": {
                "fed_policy": "Analysis needed",
                "economic_data": "Analysis needed",
                "global_events": "Analysis needed"
            },
            "sector_trends": {
                "technology": {
                    "ai_developments": "Analysis needed",
                    "chip_shortage": "Analysis needed",
                    "cloud_computing": "Analysis needed"
                },
                "regulatory": {
                    "antitrust": "Analysis needed",
                    "data_privacy": "Analysis needed",
                    "ai_regulation": "Analysis needed"
                }
            }
        },
        "symbol_data": fetch_market_context(symbols)
    }
    
    # Save report
    filename = f"market_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nMarket report saved to {filename}")
    return filename

if __name__ == "__main__":
    save_market_report()
