"""
Generate and save trading signals for our universe of symbols.
"""

import json
from datetime import datetime
from market_analyzer import fetch_market_data, MarketData, SYMBOLS

def analyze_symbol(market_data: MarketData) -> dict:
    """Generate trading signals and position sizing for a symbol"""
    
    # Calculate trend direction
    trend = "bullish" if (market_data.close > market_data.sma_20 > market_data.sma_50) else \
            "bearish" if (market_data.close < market_data.sma_20 < market_data.sma_50) else \
            "neutral"
    
    # Calculate momentum
    momentum = "bullish" if market_data.rsi_14 > 50 and market_data.macd['hist'] > 0 else \
               "bearish" if market_data.rsi_14 < 50 and market_data.macd['hist'] < 0 else \
               "neutral"
    
    # Determine signal strength (0 to 1)
    signal_strength = 0.0
    if trend == momentum:  # Trend and momentum align
        signal_strength = min(abs(market_data.rsi_14 - 50) / 30, 1.0)  # Scale RSI distance from neutral
    
    # Calculate position size based on volatility and signal strength
    # Use ATR for volatility-based position sizing
    account_risk = 0.02  # 2% max risk per trade
    risk_multiple = 2.0  # Risk 2 ATRs
    stop_distance = market_data.atr_14 * risk_multiple
    
    # Adjust position size based on signal strength
    position_size = (account_risk * signal_strength) if signal_strength > 0 else 0
    
    # Calculate entry and exit prices
    if trend == "bullish":
        entry = market_data.close
        stop_loss = entry - stop_distance
        target = entry + (stop_distance * 1.5)  # 1.5:1 reward-to-risk
    elif trend == "bearish":
        entry = market_data.close
        stop_loss = entry + stop_distance
        target = entry - (stop_distance * 1.5)  # 1.5:1 reward-to-risk
    else:
        entry = stop_loss = target = market_data.close
    
    return {
        "symbol": market_data.symbol,
        "timestamp": market_data.timestamp,
        "price": market_data.close,
        "signal": {
            "direction": trend,
            "momentum": momentum,
            "strength": signal_strength,
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target
        },
        "position_sizing": {
            "account_risk_pct": position_size * 100,
            "atr": market_data.atr_14,
            "stop_distance": stop_distance
        },
        "technical_levels": {
            "sma_20": market_data.sma_20,
            "sma_50": market_data.sma_50,
            "sma_200": market_data.sma_200,
            "rsi": market_data.rsi_14,
            "macd_hist": market_data.macd['hist'],
            "bb_upper": market_data.bb_bands['upper'],
            "bb_lower": market_data.bb_bands['lower']
        },
        "fundamentals": {
            "market_cap": market_data.market_cap,
            "sector": market_data.sector,
            "beta": market_data.beta,
            "pe_ratio": market_data.pe_ratio
        }
    }

def main():
    """Generate and save trading signals for all symbols"""
    signals = []
    
    print(f"\nAnalyzing {len(SYMBOLS)} symbols...")
    for symbol in SYMBOLS:
        try:
            print(f"\nProcessing {symbol}...")
            market_data = MarketData.from_json(fetch_market_data(symbol))
            signal = analyze_symbol(market_data)
            signals.append(signal)
            
            # Print summary
            print(f"Signal: {signal['signal']['direction'].upper()}")
            print(f"Strength: {signal['signal']['strength']:.2%}")
            print(f"Position Size: {signal['position_sizing']['account_risk_pct']:.2f}%")
            print(f"Entry: ${signal['signal']['entry']:.2f}")
            print(f"Stop: ${signal['signal']['stop_loss']:.2f}")
            print(f"Target: ${signal['signal']['target']:.2f}")
            
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
    
    # Save signals to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trading_signals_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "signals": signals
        }, f, indent=2)
    
    print(f"\nSignals saved to {filename}")

if __name__ == "__main__":
    main()
