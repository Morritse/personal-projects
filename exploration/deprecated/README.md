# Quantum Trading Platform

A sophisticated trading platform that uses an ensemble of technical indicators across multiple timeframes to generate trading signals for the S&P500 ETF (SPY).

## Features

- Multiple timeframe analysis with appropriate historical data:
  * 5-minute bars: ~2400 bars (1 month of market hours)
  * 15-minute bars: ~800 bars
  * 1-hour bars: 200 bars
- Comprehensive technical indicator coverage using TA-Lib
- Ensemble decision making with weighted indicator groups
- Real-time data fetching using Alpaca API
- Normalized indicator signals for consistent scoring
- Position tracking and trade signal generation

## Technical Indicators

The platform uses multiple groups of indicators:

1. Overlap Studies (30% weight)
   - Moving averages (SMA, EMA, DEMA, TEMA)
   - Bollinger Bands
   - MAMA, KAMA, and more

2. Momentum Indicators (30% weight)
   - RSI, MACD, Stochastic
   - ADX, CCI, MFI
   - ROC, Williams %R, and more

3. Volume Indicators (15% weight)
   - On Balance Volume
   - Chaikin A/D Line
   - A/D Oscillator

4. Volatility Indicators (15% weight)
   - Average True Range
   - Normalized ATR
   - True Range

5. Cycle Indicators (10% weight)
   - Hilbert Transform indicators
   - Dominant Cycle Period
   - Sine Wave

## Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your Alpaca account:
   - Create an account at [Alpaca](https://app.alpaca.markets)
   - Get your API keys from the dashboard
   - Copy `.env.template` to `.env` and fill in your API keys

3. Configure the strategy (optional):
   - Adjust indicator weights in `config.py`
   - Modify timeframes and thresholds
   - Add or remove indicators from groups

## How It Works

The platform operates on a one-minute cycle:

1. Efficient Data Collection (Every Minute):
   - Fetches only 5-minute bars (2400 bars for 1 month of market hours)
   - Automatically aggregates into larger timeframes:
     * 5-minute bars → Base data
     * 15-minute bars → Aggregated from three 5-min bars
     * 60-minute bars → Aggregated from twelve 5-min bars
   - Single API call reduces rate limiting and ensures data consistency

2. Indicator Calculations:
   - For each timeframe, calculates all technical indicators using their required lookback periods
   - Examples:
     * 200-day MA uses 200 bars of the hourly data
     * RSI-14 uses 14 bars of each timeframe
     * MACD uses 26, 12, and 9 periods of each timeframe

3. Signal Generation (Every Minute):
   - Each indicator produces a normalized signal (-1 to +1)
   - Signals are grouped by type (momentum, trend, volume, etc.)
   - Each timeframe's signals are weighted (5min: 50%, 15min: 30%, 1hr: 20%)
   - Final ensemble score combines all weighted signals

4. Risk Management & Position Sizing:
   - Position size limited to 2% of account equity
   - Automatic stop loss orders at 2% below entry
   - Maximum risk per trade of 1% of equity
   - Dynamic position sizing based on current price
   - Continuous position monitoring and P&L tracking

5. Trading Decisions:
   - Score > 0.7: Generate BUY signal with position sizing
   - Score < -0.7: Generate SELL signal and close position
   - Otherwise: HOLD current position

Run the strategy:
```bash
python main.py
```

## Output Format

The strategy outputs:
```
=== 2024-01-01 10:00:00 ===
Decision: BUY
Ensemble Score: 0.750

Indicator Group Signals:
  overlap: 0.823
  momentum: 0.654
  volume: 0.912
  volatility: 0.445
  cycle: 0.534

Active Indicators by Timeframe:
  short: 42 indicators
  medium: 42 indicators
  long: 42 indicators

Position Information:
  Quantity: 100 shares
  Entry Price: $594.50
  Current Price: $594.79
  Market Value: $59,479
  Unrealized P&L: $29.00 (0.049%)
  Stop Loss: $582.61
```

## Architecture

- `config.py`: Configuration settings and parameters
- `data_fetcher.py`: Handles market data retrieval from Alpaca
- `indicators.py`: Technical indicator calculations using TA-Lib
- `strategy.py`: Ensemble strategy and decision making
- `main.py`: Entry point and output formatting

## Risk Management

The platform implements comprehensive risk management:

1. Position Sizing:
   - Maximum 2% of equity per position
   - Dynamic sizing based on current price
   - Minimum 1 share requirement

2. Stop Loss Management:
   - Automatic stop loss orders
   - 2% stop loss from entry price
   - Good-till-cancelled orders

3. Account Protection:
   - Maximum 1% risk per trade
   - Position tracking and P&L monitoring
   - Day trade counting and pattern day trading protection

4. Error Handling:
   - Order verification and logging
   - Position reconciliation
   - Market hours awareness
   - API rate limit management

## Backtesting Framework

The platform includes a sophisticated backtesting framework for strategy optimization:

1. Core Components:
   - Historical data simulation with realistic trading
   - Parameter optimization using Bayesian search
   - Performance analytics and visualization
   - Multi-timeframe support

2. Features:
   - Time series cross-validation
   - Position sizing based on signal strength
   - Transaction cost modeling
   - Detailed performance metrics
   - Parameter importance analysis
   - Equity curve visualization

3. Parameter Optimization:
   - Optimizes signal weights and thresholds
   - Uses Bayesian optimization for efficiency
   - Cross-validates across time periods
   - Prevents overfitting through validation

4. Usage:
   ```python
   # Run example backtest
   cd backtesting
   python example.py
   ```

   See `backtesting/README.md` for detailed documentation.

5. Key Metrics:
   - Sharpe Ratio
   - Win Rate
   - Maximum Drawdown
   - Return Metrics
   - Trade Statistics

## Notes

- Paper trading by default for safe testing
- Optimized data collection with single API call
- Historical data aggregated efficiently
- Normalized indicators (-1 to +1 range)
- Multi-timeframe weighting (50% 5min, 30% 15min, 20% 1hr)
- Comprehensive position and risk tracking
- Error handling and logging included
- Market hours awareness
- API rate limit optimization

## Disclaimer

This is a demonstration platform. Always backtest trading strategies thoroughly before using them with real money. Past performance does not guarantee future results.
