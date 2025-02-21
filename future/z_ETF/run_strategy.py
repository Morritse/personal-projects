import os
import pandas as pd
import numpy as np
from typing import Dict, Any
from fetch_working_data import fetch_data, ETF_SYMBOLS
from params import MA_PARAMS, TRADING_PARAMS  # Import both parameter sets

def compute_signals(df: pd.DataFrame, ma_params: Dict[str, Any], trading_params: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute trading signals using MA crossover and volatility targeting.
    """
    df = df.copy()
    
    # Get MA parameters
    lookback_fast = ma_params['lookback_fast']
    lookback_slow = ma_params['lookback_slow']
    vol_lookback = ma_params['vol_lookback']
    vol_target = ma_params['vol_target']
    adx_threshold = ma_params['adx_threshold']
    
    # Calculate returns and volatility
    df['returns'] = df['close'].pct_change()
    df['roll_std'] = df['returns'].rolling(window=vol_lookback).std() * np.sqrt(252)
    df['roll_std'] = df['roll_std'].replace(0, np.nan)
    
    # Moving averages
    df['ma_fast'] = df['close'].rolling(window=lookback_fast).mean()
    df['ma_slow'] = df['close'].rolling(window=lookback_slow).mean()
    
    # Basic MA signal
    df['signal'] = 0
    df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1
    df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1
    
    # ADX Filter (if threshold provided)
    if adx_threshold is not None:
        # Calculate ADX
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR and DM
        tr_smooth = tr.rolling(window=14).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=14).mean() / tr_smooth
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14).mean() / tr_smooth
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=14).mean()
        
        # Apply ADX filter
        df['adx'] = adx
        df.loc[df['adx'] < adx_threshold, 'signal'] = 0
    
    # Volatility targeting for position sizing
    df['pos_size'] = vol_target / (df['roll_std'] + 1e-9)
    df['pos_size'] = df['pos_size'].clip(upper=1.0)
    
    # Final position (before shifting)
    df['position'] = df['signal'] * df['pos_size']
    
    # Shift position for next-day execution
    df['position_next'] = df['position'].shift(1)
    
    return df

def main():
    print("Fetching latest data...")
    data_dict = {}
    
    for symbol in ETF_SYMBOLS:
        print(f"Fetching {symbol}...")
        df = fetch_data(symbol)
        if df is not None and not df.empty:
            # Rename timestamp to Date and set as index
            df.rename(columns={'timestamp': 'Date'}, inplace=True)
            df.set_index('Date', inplace=True)
            # Convert UTC to local time and remove timezone info
            df.index = df.index.tz_convert(None)
            data_dict[symbol] = df
    
    print("\nComputing signals for each symbol...")
    signals_dict = {}
    
    for symbol, df in data_dict.items():
        print(f"\nProcessing {symbol}:")
        
        try:
            # Compute signals
            df_signals = compute_signals(df, MA_PARAMS, TRADING_PARAMS)
            
            # Get latest position
            latest_pos = df_signals['position_next'].iloc[-1]
            latest_date = df_signals.index[-1]
            
            print(f"  Latest date: {latest_date.date()}")
            print(f"  Position for next day: {latest_pos:.2f}")
            
            # Store results
            signals_dict[symbol] = df_signals
            
            # Additional analytics
            n_trades = (df_signals['position'] != df_signals['position'].shift(1)).sum()
            avg_pos = df_signals['position'].abs().mean()
            print(f"  Total trades: {n_trades}")
            print(f"  Average position size: {avg_pos:.2f}")
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    print("\nSaving positions...")
    # Create dictionary of latest positions
    positions = {}
    for symbol, df in signals_dict.items():
        positions[symbol] = df['position_next'].iloc[-1]
    
    # Save positions to JSON
    import json
    positions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions.json")
    with open(positions_file, 'w') as f:
        json.dump(positions, f, indent=2)
    print(f"Saved positions to {positions_file}")

if __name__ == "__main__":
    main()
