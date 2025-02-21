##############################################################################
# config.py
##############################################################################

config = {
    
    "allow_shorts": [False],
    
    # Single-element lists for your parameters
    "Risk Per Trade": [0.03],
    "Min Stop Pct":   [0.01],
    "Max Stop Pct":   [0.04],

    # Windows in single-element lists
    "MFI Period":      [14],
    "VWAP Window":     [25],
    "ATR Period":      [14],
    "Current Window":  [125],
    "Historical Window":[1000],

    # Trend thresholds
    'trend_params': {
        'ema_short_span': [40],
        'ema_long_span':  [70],
        'min_trend_strength': [0.003]
    },

    # MFI thresholds
    'mfi_entry': {
        "bull": [65],
        "bear": [30]
    },
    'mfi_exit': {
        "bull": [85],
        "bear": [20]
    },

    # Volatility multiplier
    'Volatility Multiplier': [0.2],

    # Minute confirmation
    'minute_confirmation': {
        "min_volume_increases": [2],
        "max_spread_pct":       [0.006],
        "max_price_volatility": [0.003]
    },

    # Regime parameters
    'regime_params': {
        'bear_high_vol': {
            "position_scale": [2.0],
            "reward_risk":    [2.0],
            "stop_mult":      [1.25],
            'trailing_stop':  [True]
        },
        'bull_high_vol': {
            "position_scale": [2.75],
            "reward_risk":    [1.25],
            "stop_mult":      [1.25],
            'trailing_stop':  [True]
        }
    },

    # Symbol list for convenience
    'SYMBOLS': [
        'BTC/USD','ETH/USD','LTC/USD','BCH/USD','LINK/USD',
        'UNI/USD','AAVE/USD','SOL/USD','DOGE/USD','DOT/USD',
        'AVAX/USD','SUSHI/USD'
    ]
}
SYMBOLS = [
    'BTC/USD', 'ETH/USD', 'LTC/USD', 'BCH/USD', 
    'LINK/USD', 'UNI/USD', 'AAVE/USD', 'SOL/USD', 
    'DOGE/USD', 'DOT/USD', 'AVAX/USD', 'SUSHI/USD'
]
