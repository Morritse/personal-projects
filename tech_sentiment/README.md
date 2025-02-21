# AI Trader

Automated trading system that combines social sentiment and technical analysis to identify and trade trending stocks.

## Strategy

### Entry Criteria
1. Stock appears in top 10 by combined ranking:
   - Get sentiment ranks (1-20) from both sources:
     * ApeWisdom: Ranked by mentions and sentiment
     * Stocktwits: Ranked by bullish ratio * log(mentions)  (requires >60% bullish)
   - Take best rank between sources (or average if in both)
   - Get technical rank (1-N) based on:
     * Moving averages
     * MACD
     * RSI
     * Momentum
   - Final rank = sentiment_rank + technical_rank
   - Lower is better (e.g. rank 1 sentiment + rank 1 technicals = best score of 2)
2. Technical score > 40%
3. Not already in portfolio
4. Places 8% position size order

### Exit Criteria
Exits position if any 2 of these signals occur:
1. Price below both moving averages
2. Strong bearish MACD
3. >5% loss in 24h

Also exits if:
- Falls out of top 10 AND technical score < 40%

### Position Management
- 8% position size per stock
- Maximum 10 positions (80% total exposure)
- Monitors every 5 minutes for:
  * Exit signals
  * New opportunities
  * Technical score changes

## Setup

1. Install TA-Lib (required for technical analysis):
   - On Ubuntu/Debian:
   ```bash
   sudo apt-get install ta-lib
   ```
   - On macOS:
   ```bash
   brew install ta-lib
   ```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Create an Alpaca paper trading account:
   - Sign up at https://app.alpaca.markets/signup
   - Go to Paper Trading section
   - Get your API key and secret

4. Create .env file with your Alpaca credentials:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

Note: The system uses paper trading by default for safety. Make sure you're using paper trading API keys, not live trading keys.

## Usage

Run the trading loop:
```bash
python run_trader.py
```

This will:
1. Check existing positions every 5 minutes
2. Exit positions that trigger signals
3. Enter new positions from top 10 list
4. Queue orders for market open if after hours

## Components

- `social_scanner.py`: Gets and ranks stocks from:
  * ApeWisdom (Reddit sentiment)
  * Stocktwits (watchers + sentiment)
- `technical_analysis.py`: Technical indicators
- `position_manager.py`: Order execution and position tracking
- `trader.py`: Main trading logic
- `run_trader.py`: Trading loop with 5-minute cycle

## Risk Warning

This is experimental software for educational purposes. Use paper trading to test strategies before considering real money. The system implements basic risk management through position sizing and technical exit signals, but you should adjust parameters based on your risk tolerance.
