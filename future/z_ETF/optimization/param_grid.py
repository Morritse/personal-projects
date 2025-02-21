# config.py

PARAM_GRID = {
    # --- Moving Average Parameters ---
    "lookback_fast":  [10, 20, 50, 100],    # short/fast MA periods
    "lookback_slow":  [50, 100, 200, 300],  # long/slow MA periods

    # --- Volatility Targeting / Sizing ---
    "vol_lookback":   [20, 60, 120],        # for calculating rolling volatility
    "vol_target":     [0.10, 0.15, 0.20],   # annualized volatility target

    # --- Core Risk Management ---
    "stop_atr_multiple": [1.5, 2.0, 2.5, 3.0],  # base ATR stop multiple
    "partial_exit_1":    [0.75, 1.0, 1.25],     # partial exit #1 ATR threshold
    "partial_exit_2":    [1.5, 2.0],            # partial exit #2 ATR threshold

    # --- Time-based & Trailing Exits ---
    "time_stop":           [0, 10, 20, 30],     # 0 = disabled, or # of days in trade
    "trailing_stop_factor": [0.0, 1.5, 2.0, 3.0],  # 0.0 = no trailing stop

    # --- ADX Filter (optional) ---
    "adx_threshold": [None, 10, 20],

    # --- Scaling / Correlation ---
    "scaling_mode": ["none", "pyramid"],  
    "corr_filter":  [False, 0.60, 0.80], 

    # --- Debug / etc. ---
    "debug": [False]
}
