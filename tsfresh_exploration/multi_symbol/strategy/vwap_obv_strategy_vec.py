import pandas as pd
import numpy as np

class VWAPOBVCrossoverVec:
    def __init__(self, params):
        # VWAP parameters
        self.vwap_length = params.get('vwap_length', 50)
        
        # MFI parameters
        self.mfi_length = params.get('mfi_length', 9)
        self.mfi_oversold = params.get('mfi_oversold', 30)
        self.mfi_overbought = params.get('mfi_overbought', 70)
        
        # Regime parameters
        self.regime_window = params.get('regime_window', 20)
        self.vol_percentile = params.get('vol_percentile', 67)
        
        # Position sizing
        self.size_factor = params.get('size_factor', 1.0)
        self.max_position = params.get('max_position', 0.25)
        
        # Exit rules
        self.max_hold_hours = params.get('max_hold_hours', 24)
        self.profit_target = params.get('profit_target', 0.02)
        self.stop_loss = params.get('stop_loss', 0.02)
    
    def calculate_vwap(self, df, length):
        """Calculate VWAP using vectorized operations."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).rolling(length).sum() / df['volume'].rolling(length).sum()
    
    def calculate_mfi(self, df, length):
        """Calculate Money Flow Index using vectorized operations."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        # Calculate positive and negative money flow
        price_change = typical_price.diff()
        
        # Initialize flows as float64
        positive_flow = pd.Series(0.0, index=df.index)
        negative_flow = pd.Series(0.0, index=df.index)
        
        # Use numpy where for type-safe assignment
        positive_flow = np.where(price_change > 0, raw_money_flow, 0)
        negative_flow = np.where(price_change < 0, raw_money_flow, 0)
        
        # Convert to series
        positive_flow = pd.Series(positive_flow, index=df.index)
        negative_flow = pd.Series(negative_flow, index=df.index)
        
        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=length).sum()
        negative_mf = negative_flow.rolling(window=length).sum()
        
        # Handle division by zero
        money_flow_ratio = np.where(negative_mf != 0, 
                                  positive_mf / negative_mf,
                                  np.inf)
        
        return 100 - (100 / (1 + money_flow_ratio))
    
    def calculate_obv(self, df):
        """Calculate On-Balance Volume using vectorized operations."""
        close_change = df['close'].diff()
        
        # Use numpy where for type-safe volume adjustment
        volume = df['volume'].values
        adjusted_volume = np.where(close_change > 0, volume,
                                 np.where(close_change < 0, -volume, 0))
        
        return pd.Series(adjusted_volume, index=df.index).cumsum()
    
    def determine_regime(self, df):
        """Determine market regime using vectorized operations."""
        returns = df['close'].pct_change()
        vol = returns.rolling(window=self.regime_window).std() * np.sqrt(252)
        ret = returns.rolling(window=self.regime_window).mean() * 252
        
        # Calculate rolling volatility threshold
        vol_threshold = vol.rolling(window=self.regime_window).quantile(self.vol_percentile/100)
        
        # Create regime series
        regime = pd.Series('none', index=df.index)
        bull_mask = (ret > 0) & (vol > vol_threshold)
        bear_mask = (ret <= 0) & (vol > vol_threshold)
        
        regime[bull_mask] = 'bull_high_vol'
        regime[bear_mask] = 'bear_high_vol'
        
        return regime
    
    def calculate_position_sizes(self, df):
        """Calculate position sizes using vectorized operations."""
        returns = df['close'].pct_change()
        vol = returns.rolling(window=self.regime_window).std() * np.sqrt(252)
        
        # Base size on inverse of volatility
        vol_size = 1 / (vol * np.sqrt(252))
        
        # Scale by size factor and cap at max position
        size = vol_size * self.size_factor
        return np.clip(size, 0.001, self.max_position)
    
    def run(self, df):
        """Run strategy using vectorized operations."""
        # Calculate indicators
        df = df.copy()
        df['vwap'] = self.calculate_vwap(df, self.vwap_length)
        df['mfi'] = self.calculate_mfi(df, self.mfi_length)
        df['obv'] = self.calculate_obv(df)
        df['obv_change'] = df['obv'].diff()
        df['regime'] = self.determine_regime(df)
        df['position_size'] = self.calculate_position_sizes(df)
        
        # Generate entry signals
        entry_signals = (
            (df['regime'] != 'none') &  # High vol regime
            (df['mfi'] < self.mfi_oversold) &  # Oversold MFI
            (df['close'] < df['vwap']) &  # Price below VWAP
            (df['obv_change'] < 0)  # Falling OBV
        )
        
        # Generate exit signals
        exit_signals = (
            (df['close'] > df['vwap']) |  # Price above VWAP
            (df['mfi'] > self.mfi_overbought)  # Overbought MFI
        )
        
        # Initialize position tracking
        df['position'] = 0.0
        df['entry_price'] = 0.0
        df['entry_time'] = pd.NaT
        df['exit_reason'] = ''
        
        # Process signals
        trades = []
        position = 0
        entry_price = 0
        entry_time = None
        
        for i in range(len(df)):
            if position == 0 and entry_signals.iloc[i]:
                # Enter position
                position = df['position_size'].iloc[i]
                entry_price = df['close'].iloc[i]
                entry_time = df.index[i]
                
                trades.append({
                    'timestamp': df.index[i],
                    'action': 'BUY',
                    'price': entry_price,
                    'size': position,
                    'regime': df['regime'].iloc[i]
                })
            
            elif position > 0:
                # Check time-based exit
                if entry_time:
                    hours_held = (df.index[i] - entry_time).total_seconds() / 3600
                    time_exit = hours_held >= self.max_hold_hours
                
                # Check profit target and stop loss
                returns = (df['close'].iloc[i] - entry_price) / entry_price
                profit_exit = returns >= self.profit_target
                stop_exit = returns <= -self.stop_loss
                
                # Determine exit
                should_exit = exit_signals.iloc[i] or time_exit or profit_exit or stop_exit
                
                if should_exit:
                    exit_price = df['close'].iloc[i]
                    pnl = (exit_price - entry_price) * position * 100000  # Assuming $100k capital
                    
                    # Determine exit reason
                    if time_exit:
                        exit_reason = 'time_exit'
                    elif profit_exit:
                        exit_reason = 'profit_target'
                    elif stop_exit:
                        exit_reason = 'stop_loss'
                    else:
                        exit_reason = 'signal'
                    
                    trades.append({
                        'timestamp': df.index[i],
                        'action': 'SELL',
                        'price': exit_price,
                        'pnl': pnl,
                        'regime': df['regime'].iloc[i],
                        'exit_reason': exit_reason
                    })
                    
                    position = 0
                    entry_price = 0
                    entry_time = None
        
        return trades
