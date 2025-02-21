# config = {
#     # Strategy parameters
#     'Regime Window': [30],
#     'Volatility Percentile': [50],  # Lower threshold to detect more regimes
#     'VWAP Window': [20],   
#     'mfi_entry': [30],
#     'bear_exit': [60],
#     'bull_exit': [65],
#     'MFI Period': [9],


#     # Portfolio parameters
#     'Initial Capital': [100000],
#     'Max Positions': [5],  # Test different position limits
#     'Max Allocation': [0.20],  # 20% max per position
#     'Max Portfolio Risk': [0.02],  # 2% portfolio risk
    

#     'stop_mult': [1.2],
#     'reward_risk': [1.5],
#     'bear_position_scale': [1.5],
#     'bull_position_scale': [2.0],


#     'Risk Per Trade': [0.02],

#     'bear_trailing_stop': [True],
#     'bull_trailing_stop': [True],
    
#     # Regime parameters
#     'regime_params': {
#         'bear_high_vol': {},
#         'bull_high_vol': {}
#     }
# }

config = {
    # Portfolio parameters
    'Initial Capital': [50000],
    'Max Positions': [5],  # Test different position limits
    'Max Allocation': [0.20],  # 20% max per position
    'Max Portfolio Risk': [0.02],  # 2% portfolio risk
    
    # Strategy parameters
    'Regime Window': [15],
    'Volatility Percentile': [67],
    'VWAP Window': [30],   

    'mfi_entry': [35],
    'bear_exit': [60,],
    'bull_exit': [90],
    'MFI Period': [9],
    'Risk Per Trade': [0.02],

    'stop_mult': [2.0],
    'reward_risk': [3.0],
    'bear_position_scale': [3.0],
    'bull_position_scale': [2.0],

    'bear_trailing_stop': [True],
    'bull_trailing_stop': [True],
    
    # Regime parameters
    'regime_params': {
        'bear_high_vol': {},
        'bull_high_vol': {}
    }
}
