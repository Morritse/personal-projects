# Candlesticker Trading System

A comprehensive market analysis and trading platform that combines technical analysis, sentiment analysis, and specialized market scanners for identifying trading opportunities.

## Overview

This system integrates multiple analysis approaches to identify and execute trading opportunities across different market sectors. It features real-time market analysis, sentiment tracking, and automated trading capabilities.

## Components

### Market Analysis
- `market_analyzer.py`: Core market analysis engine
- `market_report.py`: Generates detailed market reports
- `technical_analysis.py`: Technical indicator analysis

### Scanners
- `gas_scanner.py`: Energy sector opportunity scanner
- `tech_scanner.py`: Technology sector analysis
- `opportunity_scanner.py`: General market opportunity detection

### Trading Implementation
- `alpaca_trader.py`: Live trading via Alpaca API
- `paper_trader.py`: Paper trading simulation
- `moo_trader.py`: Market-on-open trading strategy
- `trading_signals.py`: Signal generation system

### Sentiment Analysis
- `sentiment_analyzer.py`: Market sentiment analysis
- Processes news, social media, and market data

## Features

### Market Analysis
- Real-time technical analysis
- Multi-timeframe analysis
- Volume profile analysis
- Price action patterns
- Market sector rotation

### Trading Capabilities
- Automated trading execution
- Paper trading simulation
- Multiple broker integration
- Risk management system
- Position sizing logic

### Opportunity Detection
- Sector-specific scanners
- Technical pattern recognition
- Volume anomaly detection
- Sentiment-based opportunities
- Market regime detection

### Reporting
- Automated market reports
- Trading opportunity alerts
- Performance analytics
- Risk metrics tracking
- Portfolio analysis

## Data Sources

The system analyzes data from multiple sources:
- Market price data
- Volume information
- News feeds
- Social media sentiment
- Technical indicators

## Configuration

Trading parameters can be configured in `trading_plan.json`:
- Risk parameters
- Position sizing rules
- Entry/exit conditions
- Scanner thresholds
- Analysis timeframes

## Usage

1. Run market analysis:
```bash
python market_analyzer.py
```

2. Generate market report:
```bash
python market_report.py
```

3. Start opportunity scanner:
```bash
python opportunity_scanner.py
```

4. Execute trading strategy:
```bash
python alpaca_trader.py  # For live trading
python paper_trader.py   # For paper trading
```

## Requirements

- Python 3.8+
- Alpaca API credentials
- Market data access
- News API access
- Technical analysis libraries

## Risk Management

- Position size limits
- Stop-loss implementation
- Profit taking rules
- Portfolio diversification
- Market exposure limits

## Output Files

The system generates various JSON output files:
- Market opportunities reports
- Sentiment analysis results
- Technical analysis data
- Trading signals
- Performance metrics

## Future Enhancements

1. Additional market scanners
2. Enhanced sentiment analysis
3. Machine learning integration
4. Real-time alerts system
5. Extended broker integration
