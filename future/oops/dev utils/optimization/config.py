
sMA_GRID = {
    'lookback_fast': [50, 100, 125],
    'lookback_slow': [200, 250],
    'vol_lookback': [20, 40],
    'vol_target': [0.15, 0.20],
    'stop_atr_multiple': [1.5, 2.0],
    'debug': [False]
}

MA_GRID = {
    'lookback_fast': [95],
    'lookback_slow': [200],
    'vol_lookback': [20],
    'vol_target': [0.15],
    'stop_atr_multiple': [1.8],
    'debug': [False]
}



STRADING_STRATEGY_GRID = {
    'partial_exit_1': [0, 0.75, 1.0],
    'partial_exit_2': [0, 1.5, 2.0],
    'time_stop': [0, 10, 30],
    'trailing_stop_factor': [0, 2.0, 3.0], 
    'adx_threshold': [None, 20, 25],
    'scaling_mode': ['none', 'pyramid'],
    'corr_filter': [False, 0.6, 0.8]
}

# config.py snippet for the "optimized" parameters

TRADING_STRATEGY_GRID = {
    'partial_exit_1': [1],
    'partial_exit_2': [1.0],
    'time_stop': [22],
    'trailing_stop_factor': [2.0],
    'adx_threshold': [10],
    'scaling_mode': ['none'],
    'corr_filter': [.7]
}
