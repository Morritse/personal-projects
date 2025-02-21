# best_config.py
"""
Configuration file storing the best parameters found so far for both
the MA-based strategy logic and the trading/execution logic.
"""


MA_PARAMS = {
    "lookback_fast":  20,
    "lookback_slow":  200,
    "vol_lookback":   60,
    "vol_target":     0.15,
    "adx_threshold":  None  
}

TRADING_PARAMS = {
    "partial_exit_1":       1.0,

    "partial_exit_2":       1.0,

    "time_stop":            30,

    "trailing_stop_factor": 0.0,

    "scaling_mode":         "none",
}


