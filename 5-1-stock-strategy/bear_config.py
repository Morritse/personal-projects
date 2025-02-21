bear_config = {
    # --- Portfolio parameters ---
    "Initial Capital": [100000],
    "Max Positions": [4],
    "Max Allocation": [.3],       # 30% max per position
    "Max Portfolio Risk": [.04],   # 4% portfolio risk
    
    # --- High-level Strategy parameters ---
    "Regime Window": [20],           # e.g., 30 bars for smoothed returns
 #   "Volatility Percentile": [65],   # threshold for "high vol"
    "Volatility Percentile": [10], 
    "VWAP Window": [15],             # e.g., 20 x 5-min bars => ~100 minutes

    # Bear-specific parameters for shorting
    "entry_mfi": [55],               # MFI overbought threshold for short entries
    "exit_mfi": [15],                # MFI oversold threshold for short exits
    "MFI Period": [9],
    "Risk Per Trade": [.005],      # 0.75% per trade risk
    "stop_mult": [2],              # ATR multiplier for stops
    "reward_risk": [1],            # Reward:Risk ratio
    "position_scale": [3.5],         # Position size multiplier
    
    # Slippage / trailing stops
    "trailing_stop": [True],         # Enable trailing stops
    "market_slippage": [0.02],       # 2 cents slippage
    
    # Technical conditions
    "vwap_entry": ["below"],         # Enter shorts when price breaks below VWAP
    "obv_condition": ["falling"],     # Enter when OBV is falling (distribution)
    
    # Time-based parameters
    "max_hold_hours": [5],           # Maximum position hold time
    "min_hold_hours": [0],           # Minimum hold before technical exit checks
}
