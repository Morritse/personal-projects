import pandas as pd
import numpy as np

class VWAPOBVCrossover:
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
        
        # State variables
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.regime = 'none'
    
    def calculate_vwap(self, df, length):
        """Calculate VWAP."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(length).sum() / df['volume'].rolling(length).sum()
        return vwap
    
    def calculate_mfi(self, df, length):
        """Calculate Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']
        
        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)
        
        # Calculate positive and negative money flow
        price_change = typical_price - typical_price.shift(1)
        positive_flow[price_change > 0] = raw_money_flow[price_change > 0]
        negative_flow[price_change < 0] = raw_money_flow[price_change < 0]
        
        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=length).sum()
        negative_mf = negative_flow.rolling(window=length).sum()
        
        money_flow_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi
    
    def calculate_obv(self, df):
        """Calculate On-Balance Volume."""
        obv = pd.Series(0, index=df.index)
        
        # First value is just the first volume
        obv.iloc[0] = df['volume'].iloc[0]
        
        # Calculate OBV
        close_change = df['close'].diff()
        for i in range(1, len(df)):
            if close_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif close_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def determine_regime(self, df):
        """Determine market regime based on returns and volatility."""
        returns = df['close'].pct_change()
        vol = returns.rolling(window=self.regime_window).std() * np.sqrt(252)
        ret = returns.rolling(window=self.regime_window).mean() * 252
        
        # Calculate volatility threshold
        vol_threshold = vol.rolling(window=self.regime_window).quantile(self.vol_percentile/100)
        
        if ret.iloc[-1] > 0 and vol.iloc[-1] > vol_threshold.iloc[-1]:
            return 'bull_high_vol'
        elif ret.iloc[-1] <= 0 and vol.iloc[-1] > vol_threshold.iloc[-1]:
            return 'bear_high_vol'
        else:
            return 'none'
    
    def calculate_position_size(self, df):
        """Calculate position size based on volatility."""
        returns = df['close'].pct_change()
        vol = returns.rolling(window=self.regime_window).std() * np.sqrt(252)
        
        # Base size on inverse of volatility
        vol_size = 1 / (vol.iloc[-1] * np.sqrt(252))
        
        # Scale by size factor and cap at max position
        size = min(vol_size * self.size_factor, self.max_position)
        
        # Ensure size is between 0.1% and max_position
        return np.clip(size, 0.001, self.max_position)
    
    def check_exits(self, row, entry_price, entry_time):
        """Check exit conditions."""
        # Time-based exit
        hours_held = (row.name - entry_time).total_seconds() / 3600
        if hours_held >= self.max_hold_hours:
            return True, 'time_exit'
        
        # Profit target
        if (row['close'] - entry_price) / entry_price >= self.profit_target:
            return True, 'profit_target'
        
        # Stop loss
        if (row['close'] - entry_price) / entry_price <= -self.stop_loss:
            return True, 'stop_loss'
        
        return False, None
    
    def run(self, df):
        """Run strategy and return trades."""
        # Calculate indicators
        df['vwap'] = self.calculate_vwap(df, self.vwap_length)
        df['mfi'] = self.calculate_mfi(df, self.mfi_length)
        df['obv'] = self.calculate_obv(df)
        df['obv_change'] = df['obv'].diff()
        
        trades = []
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        
        for i in range(self.vwap_length, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Update regime
            self.regime = self.determine_regime(df.iloc[:i+1])
            
            # Check exits if in position
            if self.position > 0:
                should_exit, exit_reason = self.check_exits(row, self.entry_price, self.entry_time)
                if should_exit or row['close'] > df['vwap'].iloc[i] or row['mfi'] > self.mfi_overbought:
                    # Calculate PnL
                    pnl = (row['close'] - self.entry_price) * self.position * 100000  # Assuming $100k capital
                    
                    trades.append({
                        'timestamp': row.name,
                        'action': 'SELL',
                        'price': row['close'],
                        'pnl': pnl,
                        'regime': self.regime,
                        'exit_reason': exit_reason if should_exit else 'signal'
                    })
                    
                    self.position = 0
                    self.entry_price = 0
                    self.entry_time = None
            
            # Check entry conditions
            elif (self.regime != 'none' and  # Only trade in high vol regimes
                  row['mfi'] < self.mfi_oversold and  # Oversold MFI
                  row['close'] < df['vwap'].iloc[i] and  # Price below VWAP
                  row['obv_change'] < 0):  # Falling OBV
                
                # Calculate position size
                self.position = self.calculate_position_size(df.iloc[:i+1])
                self.entry_price = row['close']
                self.entry_time = row.name
                
                trades.append({
                    'timestamp': row.name,
                    'action': 'BUY',
                    'price': row['close'],
                    'size': self.position,
                    'regime': self.regime
                })
        
        return trades
