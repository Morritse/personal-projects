# JNJ Trading Strategy

Custom trading strategy for Johnson & Johnson (JNJ) based on statistical analysis of volume patterns, price action, and sector correlations.

## Statistical Basis

Analysis of JNJ revealed several significant patterns:
- Strong volume asymmetry at 1-3 hour lags (z-scores: 6.21, 7.71, 8.79)
- Predictable volume cycles
- High ranking in volatility metrics
- Strong sector correlations with specific time-of-day patterns

## Cross-Indicator Analysis

Extensive testing revealed JNJ exhibits contrarian behavior relative to traditional technical indicators:

### Traditional vs. Actual Behavior

1. VWAP Crossovers
   - Traditional: Buy above VWAP (uptrend)
   - Actual: Better returns when buying below VWAP (-0.4% to -0.5% threshold)
   - Success Rate: 52.94% win rate with contrarian approach

2. Volume Patterns
   - Traditional: Buy with rising OBV (accumulation)
   - Actual: Strong bounces after high-volume selloffs
   - Best Performance: Bear high-vol regimes (48 trades, 52.08% win rate)

3. Momentum Indicators
   - Traditional: Buy when RSI/MFI > 50 (uptrend)
   - Actual: Higher success rate buying oversold conditions
   - Optimal Ranges: RSI < 35 (bear) or < 40 (bull)

### After-Hours Behavior

1. Mean Reversion Success Rates
   - Bull High Vol: 93.99% (4H timeframe)
   - Bear High Vol: 91.67% (4H timeframe)
   - Bull Low Vol: 88.59% (4H timeframe)
   - Bear Low Vol: 85.04% (4H timeframe)

2. Time Window Analysis
   - Best Period: 16:00-19:00 ET
   - Highest Success: First 2 hours after market close
   - Average Duration: 2-4 hours for mean reversion

3. Volume Characteristics
   - Strong divergences (z-score > 2.0): 803 signals
   - Most active hours: 16:00-17:00 (173 signals)
   - Signal decay: Gradual reduction from 173 to 49 signals between 16:00-23:00

### Sector Correlation Insights

1. Time-Based Correlation
   - Strongest: 12:00-16:00 ET (0.43-0.59 correlation)
   - After-Hours: 16:00-20:00 ET (0.35-0.45 correlation)
   - Weakest: Pre-market and early morning

2. Top Correlated Symbols
   - XLV (Healthcare ETF): 0.57-0.61 correlation
   - PFE: 0.44-0.49 correlation
   - MRK: 0.41-0.45 correlation
   - ABBV: 0.39-0.41 correlation

3. Regime-Based Correlations
   - Bull High Vol: 0.59 average correlation
   - Bear High Vol: 0.53 average correlation
   - Low Vol Regimes: < 0.35 correlation

## Strategy Components

1. Volume Pattern Detection
   - Tracks 1-3 hour volume ratios
   - Calculates z-scores for pattern strength
   - Requires z-score > 2.0 for entry signals

2. Price Confirmation
   - VWAP with 3-hour window
   - 20-period EMA trend filter
   - 9-period MFI (optimized for JNJ's patterns)

3. Position Sizing
   - Base size scaled by volume pattern strength
   - Maximum 1.5x scaling for strongest patterns
   - ATR-based risk management

## Entry Conditions

1. Volume pattern z-score > 2.0
2. Price below VWAP (contrarian)
3. MFI < 30 (oversold)
4. High volatility regime
5. Strong sector correlation (> 0.4)

## Exit Conditions

1. Volume pattern weakens (z-score < 1.0)
2. Price crosses above VWAP
3. Hit take-profit (2x risk)
4. Hit stop-loss (1.5x ATR, adjusted for position size)

## Risk Management

- Dynamic stop-loss based on 3-hour ATR
- Position size scales with pattern strength
- Maximum risk per trade: 1.5x base risk
- Take-profit at 2x risk

## Regime-Based Adjustments

| Regime | Position Size | Stop Multiplier | Z-Score Threshold | Correlation Min |
|--------|--------------|-----------------|-------------------|----------------|
| Bull High Vol | 100-150% | 2.0× ATR | 2.0 | 0.4 |
| Bear High Vol | 100% | 2.0× ATR | 2.0 | 0.4 |
| Bull Low Vol | 50-75% | 1.5× ATR | 2.5 | 0.3 |
| Bear Low Vol | Skip | 1.5× ATR | 3.0 | 0.3 |

## Time-Based Filters

1. Regular Trading Hours
   - Best Window: 12:00-16:00 ET
   - Required Correlation: > 0.4
   - Volume Z-score: > 2.0

2. After-Hours Trading
   - Best Window: 16:00-19:00 ET
   - Required Correlation: > 0.35
   - Volume Z-score: > 2.5
   - Mean Reversion Success: 85-94%

3. Pre-Market/Early Morning
   - Avoid trading (weak correlations)
   - If trading, require z-score > 3.0

## Configuration

```python
strategy_configuration = {
    'Volume Pattern Threshold': 2.0,
    'Position Scale Max': 1.5,
    'MFI Period': 9,
    'EMA Period': 20,
    'VWAP Window': 180,
    'ATR Period': 3
}
```

## Dependencies
- pandas
- numpy
- talib
- matplotlib

## Usage
1. Ensure dependencies are installed
2. Configure API credentials in .env
3. Run strategy with:
```python
from strategy.vwap_obv_strategy import VWAPOBVCrossover
# Initialize and run strategy
```

## Backtest Results

### Overall Performance
- Win Rate: 52.94%
- Profit Factor: 1.21
- Average Win: $893.90
- Average Loss: $739.20

### Regime Performance
1. Bear High Vol
   - Trades: 48
   - Win Rate: 52.08%
   - Average PnL: $79.03

2. Bull High Vol
   - Trades: 3
   - Win Rate: 66.67%
   - Average PnL: $866.98

### After-Hours Performance
1. 2-Hour Timeframe
   - Success Rate: 87.83-89.51%
   - Best in Bull High Vol

2. 4-Hour Timeframe
   - Success Rate: 91.67-93.99%
   - Best in Bear High Vol

## Future Improvements
1. Dynamic correlation thresholds
2. Machine learning for regime classification
3. Integration with sector-wide volume analysis
4. Real-time news sentiment adjustment
5. Adaptive time-window selection
