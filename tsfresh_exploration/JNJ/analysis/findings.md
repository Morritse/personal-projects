 JNJ Healthcare Sector Analysis Findings

## ETF Relationships

### Correlation Patterns
- XLV shows strongest correlation (0.49-0.61)
- VHT slightly weaker but still strong (0.42-0.57)
- IYH surprisingly weak correlation (0.12-0.20)
- XLV and VHT are highly correlated (0.91)

### Component-Level Insights
- Strongest Correlations:
  * PFE (0.36-0.49)
  * MRK (0.36-0.46)
  * ABBV (0.35-0.49)
  * ABT (0.35-0.41)
- Weakest Correlations:
  * LLY (0.23-0.28)
  * TMO (0.20-0.32)
  * DHR (0.19-0.27)
- No leading indicators found (all max correlations at lag 0)

### Market Session Patterns
- Afternoon and after-hours show strongest correlations
  * After-hours: 0.57-0.59
  * Afternoon: 0.43-0.57
- Morning session surprisingly weak (0.12-0.17)
- Pre-market weakest (-0.02-0.23)
- Trading Implication: Focus on afternoon and after-hours sessions

### Volume Impact
- Base volume correlation is low (0.13 for XLV)
- High volume periods show 4-5x stronger correlation (0.63 for XLV)
- Pattern consistent across ETFs
- Trading Implication: High volume periods offer better execution opportunities

### Market Regime Analysis

#### Regime Distribution
- Bull Market, High Vol: ~28% of time
- Bull Market, Low Vol: ~24% of time
- Bear Market, High Vol: ~26% of time
- Bear Market, Low Vol: ~22% of time

#### Regime Characteristics
1. Bull Market, High Volatility
   - Strongest correlations (XLV: 0.50-0.57)
   - Best mean reversion (88-94%)
   - Top components: ABT, PFE, ABBV (0.38-0.44)
   - Ideal for aggressive trading

2. Bear Market, High Volatility
   - Strong correlations (XLV: 0.53-0.60)
   - Reliable mean reversion (88-92%)
   - Top components: PFE, MRK, BMY (0.43-0.48)
   - Good for defensive positioning

3. Bull Market, Low Volatility
   - Moderate correlations (XLV: 0.33-0.52)
   - Variable mean reversion (69-89%)
   - Top components: ABT, PFE, BMY (0.25-0.34)
   - Selective opportunities only

4. Bear Market, Low Volatility
   - Weakest correlations (XLV: 0.29-0.43)
   - Poor mean reversion (52-85%)
   - Top components: PFE, ABBV, MRK (0.19-0.33)
   - Avoid trading

#### Regime Transitions
1. High → Low Volatility:
   - Correlations weaken by ~30%
   - Mean reversion success drops 5-15%
   - Component relationships decay faster
   - Reduce position sizes

2. Low → High Volatility:
   - Correlations strengthen by ~40%
   - Mean reversion improves 10-20%
   - Component relationships tighten
   - Opportunity to scale up

3. Bull → Bear Market:
   - ETF correlations remain stable
   - Component leadership changes
   - Focus shifts to defensive names
   - Adjust component selection

### Beta Analysis
- JNJ is defensive vs all ETFs (beta < 1)
- XLV: beta ~0.72-0.78
- VHT: beta ~0.55-0.66
- IYH: very low beta ~0.05-0.07
- Trading Implication: JNJ moves about 70% as much as XLV, providing defensive characteristics

## Trading Opportunities

### Mean Reversion by Regime
1. Short-term (1H-4H):
   - Bull High Vol: 88-94% success
   - Bear High Vol: 88-92% success
   - Bull Low Vol: 89% success
   - Bear Low Vol: 81-85% success

2. Long-term (1D-1W):
   - Success rates drop significantly
   - Bull High Vol: 53-62% success
   - Bear High Vol: 53-64% success
   - Bull Low Vol: 54-69% success
   - Bear Low Vol: 52-82% success

3. Key Findings:
   - Short-term mean reversion most reliable
   - High volatility regimes offer best opportunities
   - Long-term success requires regime filtering

### Optimal Trading Conditions
1. Best Combination:
   - High volatility regime
   - Short timeframe (1-4 hours)
   - Strong volume
   - Focus on top components (PFE, MRK, ABBV)

2. Component Selection:
   - High Vol Regimes: PFE, MRK, ABBV (0.38-0.48 correlation)
   - Low Vol Regimes: ABT, PFE, BMY (0.25-0.34 correlation)
   - Avoid: LLY, TMO, DHR (consistently weak)

3. Position Sizing:
   - Full size in high vol regimes
   - Reduce by 50% in low vol
   - Exit positions in bear/low vol regime

### Risk Management
1. Afternoon sessions require wider stops due to weaker correlations
2. Low volume periods should be avoided
3. Position sizing should account for ~0.7 beta to XLV

## Trading Strategies

### 1. After-Hours Mean Reversion
- Entry: 2+ standard deviation divergence from XLV
- Timing: 16:00-24:00 ET
- Success Rate: 91-99%
- Key Advantage: Highest convergence rate, less noise
- Risk Management: Use 4-hour timeframe for confirmation

### 2. Afternoon Convergence
- Entry: Divergence during high volume
- Timing: 12:00-16:00 ET
- Success Rate: 91-94%
- Key Advantage: Good liquidity, strong correlations
- Risk Management: Scale position with volume

### 3. Volatility Regime Trading
- Entry: During high volatility periods
- Focus: XLV relationship (strongest correlation)
- Advantage: Stronger correlations (0.66 vs 0.55)
- Risk Management: Adjust position size for volatility

### 4. Component Basket Strategy
- Long/Short basket of high-correlation components:
  * Long: PFE, MRK, ABBV, ABT (0.35+ correlation)
  * Avoid: LLY, TMO, DHR (<0.30 correlation)
- Advantage: More precise exposure than ETFs
- Risk Management: Equal-weight positions

## Implementation Considerations

### Data Requirements
- 1-hour bars for all components and ETFs
- Real-time volume data for session analysis
- Volatility regime calculations (20-period rolling)
- Z-score calculations for divergence detection

### Position Sizing
- Base size on volatility regime
- Scale up during high volume periods
- Reduce size in morning session
- Maximum 2:1 leverage for mean reversion

### Risk Controls
1. Time-Based:
   - No trades first 30 minutes of session
   - Close positions before major announcements
   - Avoid holding through earnings

2. Volatility-Based:
   - Wider stops in high volatility
   - Reduce size when correlations weaken
   - Exit if volume drops significantly

3. Correlation-Based:
   - Monitor rolling correlation strength
   - Exit if correlation drops below 0.3
   - Avoid trading during correlation breakdowns

## Regime-Based Trading Implementation

### 1. Regime Identification
- Calculate 20-period rolling returns and volatility
- Compare to median volatility levels
- Use XLV as sector benchmark
- Update regime daily to avoid whipsaws

### 2. Strategy Adjustment by Regime
- Bull High Vol:
  * Focus on mean reversion
  * Use full position sizes
  * Target ABT, PFE, ABBV correlations
  * Aggressive 2-hour timeframe

- Bear High Vol:
  * Emphasize defensive components
  * Maintain full positions
  * Focus on PFE, MRK, BMY
  * Extend to 4-hour timeframe

- Bull Low Vol:
  * Selective mean reversion only
  * Reduce position sizes 50%
  * Monitor ABT leadership
  * Require higher divergence (2.5+ SD)

- Bear Low Vol:
  * Minimal trading
  * Very small positions if any
  * Focus on highest correlations only
  * Consider moving to cash

### 3. Regime Transition Handling
- High → Low Vol:
  * Scale down gradually
  * Tighten stops
  * Reduce holding periods
  * Increase divergence requirements

- Low → High Vol:
  * Scale up methodically
  * Widen stops appropriately
  * Extend holding periods
  * Standard divergence criteria

- Bull → Bear:
  * Rotate to defensive names
  * Adjust correlation expectations
  * Focus on quality of signal
  * Increase cash allocation

### 4. Risk Management by Regime
- Position Sizing:
  * High Vol: 100% of base size
  * Low Vol: 50% of base size
  * Transitions: 75% of base size

- Stop Levels:
  * High Vol: 2.0x average range
  * Low Vol: 1.5x average range
  * Always use time stops

- Correlation Minimums:
  * High Vol: > 0.4
  * Low Vol: > 0.3
  * Exit if below thresholds

## Future Investigation Areas
1. Impact of earnings announcements on correlations
2. Sector rotation effects on relationship strength
3. Pre-market vs after-hours behavior differences
4. Volume profile analysis by session
5. Correlation stability during market events
6. Component weighting optimization
7. Dynamic beta adjustment methods
8. Regime transition indicators
9. Alternative regime definitions
10. Machine learning for regime classification
