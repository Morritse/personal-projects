"""
config_ga.py

Configuration for GA-based optimization, specifying parameter ranges/bounds.
"""

# Numeric ranges: (min, max, 'type')
# For discrete lists, we define them separately.
GA_PARAM_BOUNDS = {
    # Indicator parameters
    'lookback_fast':      (20, 200, 'int'),
    'lookback_slow':      (100, 300, 'int'),
    'vol_lookback':       (10, 40, 'int'),
    'vol_target':         (0.10, 0.30, 'float'),
    'stop_atr_multiple':  (1.0, 3.0, 'float'),
    
    # Trading overlays
    'partial_exit_1':     (0.0, 1.5, 'float'),
    'partial_exit_2':     (0.0, 2.0, 'float'),
    'time_stop':          (0, 30, 'int'),
    'trailing_stop_factor': (0.0, 3.0, 'float'),
    'adx_threshold':      (0.0, 25.0, 'float'),  
}

# Categorical parameters that we pick from a list
GA_PARAM_CHOICES = {
    'scaling_mode':   ['none', 'pyramid'],
    'corr_filter':    [False, 0.6, 0.8],
    # If you like, you can allow 'None' explicitly in corr_filter if needed
    # 'corr_filter': [False, 0.6, 0.8, None]
}

# We'll fix debug=False always, or you can handle that in the GA if you prefer
DEFAULT_FIXED_PARAMS = {
    'debug': False
}
