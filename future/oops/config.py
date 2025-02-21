# config.py
"""
Configuration file storing market definitions, and optimized strategy parameters.
"""
FUTURES_SYMBOLS = {
    # Equity Index (CME / CBOT)
    "ES": {"exchange": "CME",   "name": "E-mini S&P 500"},
    "NQ": {"exchange": "CME",   "name": "E-mini Nasdaq 100"},
    "YM": {"exchange": "CBOT",  "name": "E-mini Dow"},
    "RTY": {"exchange": "CME",  "name": "E-mini Russell 2000"},
    
    # Interest Rates (CBOT)
    "ZB": {"exchange": "CBOT",  "name": "30-Year Treasury Bond"},
    "ZN": {"exchange": "CBOT",  "name": "10-Year Treasury Note"},
    "ZF": {"exchange": "CBOT",  "name": "5-Year Treasury Note"},
    
    # Energy (NYMEX)
    "CL": {"exchange": "NYMEX", "name": "Crude Oil (WTI)"},
    "NG": {"exchange": "NYMEX", "name": "Natural Gas"},
    "RB": {"exchange": "NYMEX", "name": "RBOB Gasoline"},
    "HO": {"exchange": "NYMEX", "name": "Heating Oil"},
    
    
    # Metals (COMEX)
    "GC": {"exchange": "COMEX", "name": "Gold"},
    "SI": {"exchange": "COMEX", "name": "Silver"},
    "HG": {"exchange": "COMEX", "name": "Copper"},


    # Agriculture (CBOT)
    "ZC": {"exchange": "CBOT",  "name": "Corn"},
    "ZW": {"exchange": "CBOT",  "name": "Wheat"},
    "ZS": {"exchange": "CBOT",  "name": "Soybeans"},

    # Livestock (CME)
    "LE": {"exchange": "CME",   "name": "Live Cattle"},
}


# ---------------------------------------------------------
# 2) Moving Average Strategy (Indicator) Parameters
# ---------------------------------------------------------
# Example of final chosen “best” from your grid/GA search:
MA_STRATEGY_PARAMS = {
    "lookback_fast": 100,  # Fast MA
    "lookback_slow": 200,  # Slow MA
    "vol_lookback": 20,    # Volatility lookback
    "vol_target": 0.20,    # Target annual volatility
    "stop_atr_multiple": 1.5,  # ATR-based stop
}

# ---------------------------------------------------------
# 3) Additional Trading Strategy/Exit Params
# ---------------------------------------------------------
# These might be partial exits, time stops, trailing stops, scaling mode, correlation filter, etc.
TRADING_STRATEGY_PARAMS = {
    "partial_exit_1": 0.75,  # First partial exit threshold (ATR multiple)
    "partial_exit_2": 1.0,   # Second partial exit threshold (ATR multiple)
    "time_stop": 10,         # Time-based exit in days
    "trailing_stop_factor": 2.0,  # If > stop_atr_multiple, used for trailing
    "adx_threshold": None,   # e.g. disable ADX filter
    "scaling_mode": "none",  # or "pyramid"
    "corr_filter": False,    # e.g. correlation threshold off
}

# ---------------------------------------------------------
# 4) Combined “Best” Param Set for Convenience (Optional)
# ---------------------------------------------------------
# Sometimes you might store a single dictionary that merges them all, for direct passing to the strategy:
BEST_PARAM_SET = {
    # MA strategy params
    "lookback_fast": MA_STRATEGY_PARAMS["lookback_fast"],
    "lookback_slow": MA_STRATEGY_PARAMS["lookback_slow"],
    "vol_lookback":  MA_STRATEGY_PARAMS["vol_lookback"],
    "vol_target":    MA_STRATEGY_PARAMS["vol_target"],
    "stop_atr_multiple": MA_STRATEGY_PARAMS["stop_atr_multiple"],

    # Trading strategy / exit params
    "partial_exit_1": TRADING_STRATEGY_PARAMS["partial_exit_1"],
    "partial_exit_2": TRADING_STRATEGY_PARAMS["partial_exit_2"],
    "time_stop":      TRADING_STRATEGY_PARAMS["time_stop"],
    "trailing_stop_factor": TRADING_STRATEGY_PARAMS["trailing_stop_factor"],
    "adx_threshold":  TRADING_STRATEGY_PARAMS["adx_threshold"],
    "scaling_mode":   TRADING_STRATEGY_PARAMS["scaling_mode"],
    "corr_filter":    TRADING_STRATEGY_PARAMS["corr_filter"],

    # Optionally add “debug” or other flags
    "debug": False
}

