config = {
    # Risk
    "Risk Per Trade": [.03],          # A bit lower risk per trade for intraday volatility
    "Min Stop Pct" : [.01],           # Tight stops (2%)
    "Max Stop Pct" : [.04],           # Max 5%
    
    # 1-minute windows
    "MFI Period"      : [14],         
    "VWAP Window"     : [25],          
    "ATR Period"      : [14],          
    "Current Window"  : [125],        
    "Historical Window": [1000],       
    
    # Trend thresholds
    'trend_params': {
        'ema_short_span': [40],        # 30-minute EMA (on 1-min bars)
        'ema_long_span':  [70],        # 60-minute EMA 
        'min_trend_strength': [.003]  # Slightly smaller threshold for 1-min data
    },
    
    # MFI thresholds
    'mfi_entry': {
        "bull": [65],   # Over ~65 for bullish momentum
        "bear": [30]    # Under ~35 for bearish/exhaustion
    },
    'mfi_exit': {
        "bull": [85],   # Tighter exit for overbought
        "bear": [20]    # Tighter exit for oversold
    },
    
    # Volatility multiplier
    'Volatility Multiplier': [0.2],    # Lower multiplier for 1-min data

    # Minute confirmation
    'minute_confirmation': {
        "min_volume_increases": [2],   # 2 bars of volume increases
        "max_spread_pct":       [0.006],
        "max_price_volatility": [0.003]
    },
    
    # Regime parameters
    'regime_params': {
        'bear_high_vol': {
            "position_scale": [2.0],   # Could remain higher if you want bigger short trades
            "reward_risk":    [2.0],
            "stop_mult":      [1.25],
            'trailing_stop':  [True]
        },
        'bull_high_vol': {
            "position_scale": [2.75],   # Slightly lower or same as bear
            "reward_risk":    [1.25],
            "stop_mult":      [1.25],
            'trailing_stop':  [True]
        }
    }
}

SYMBOLS = [
    'BTC/USD',
    'ETH/USD',
    'LTC/USD',
    'BCH/USD',
    'LINK/USD',
    'UNI/USD',
    'AAVE/USD',
    'SOL/USD',
    'DOGE/USD',
    'DOT/USD',
    'AVAX/USD',
    'SUSHI/USD'
]