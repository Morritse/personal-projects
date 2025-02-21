config = {
    # --- Portfolio parameters ---
    "Initial Capital": [100000],
    "Max Positions": [3],
    "Max Allocation": [.3],       # 30% max per position
    "Max Portfolio Risk": [.04],   # 4% portfolio risk
    
    # --- High-level Strategy parameters ---
    "Regime Window": [25],           # e.g., 30 bars for smoothed returns
   # "Volatility Percentile": [60],   # threshold for "high vol"
    "Volatility Percentile": [10], 
    "VWAP Window": [15],             # e.g., 20 x 5-min bars => ~100 minutes

    "mfi_entry": [35],               # MFI oversold threshold for entries
    "MFI Period": [9],
    "Risk Per Trade": [0.0075],      # 0.75% per trade risk
    "stop_mult": [3],                # ATR multiplier for stops
    "reward_risk": [1.5],            # Reward:Risk ratio
    "bull_position_scale": [3.5],    # Position size multiplier
    
    # Slippage / trailing stops
    "bull_trailing_stop": [True],    # Enable trailing stops
    
    # --- Bull regime parameters ---
    "regime_params": {
        "bull_high_vol": {
            "exit_mfi": [100],          # MFI level to exit in bull
            "stop_mult": [2.5],         # ATR multiplier for stops
            "reward_risk": [2],         # Reward:Risk ratio
            "position_scale": [3.5],    # Position size multiplier
            "trailing_stop": [True]     # Enable trailing stops
        }
    },
}
