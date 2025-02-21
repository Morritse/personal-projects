import pandas as pd
import numpy as np
from datetime import timedelta

class OptionsStrategy:
    def __init__(self, params):
        # VWAP parameters
        self.vwap_length = params.get('vwap_length', 50)
        
        # MFI parameters
        self.mfi_length = params.get('mfi_length', 9)
        self.mfi_oversold = params.get('mfi_oversold', 30)
        self.mfi_overbought = params.get('mfi_overbought', 70)
        
        # Options parameters
        self.min_dte = params.get('min_dte', 7)
        self.max_dte = params.get('max_dte', 21)
        self.strike_distance = params.get('strike_distance', 0.5)
        
        # Risk parameters
        self.profit_target = params.get('profit_target', 1.0)  # 100% profit target
        self.stop_loss = params.get('stop_loss', -0.5)  # 50% stop loss
        
        # Position sizing
        self.max_risk_pct = 0.01  # 1% account risk per trade
        self.max_position_pct = 0.05  # 5% max position size
        
        # Regime parameters
        self.regime_window = params.get('regime_window', 20)
        self.vol_percentile = params.get('vol_percentile', 67)
    
    def calculate_vwap(self, df, length):
        """Calculate VWAP using vectorized operations."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).rolling(length).sum() / df['volume'].rolling(length).sum()
            return vwap
        except Exception as e:
            print(f"Error calculating VWAP: {str(e)}")
            raise
    
    def calculate_mfi(self, df, length):
        """Calculate Money Flow Index using vectorized operations."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            raw_money_flow = typical_price * df['volume']
            
            # Calculate positive and negative money flow
            price_change = typical_price.diff()
            
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
            
            mfi = 100 - (100 / (1 + money_flow_ratio))
            return mfi
        except Exception as e:
            print(f"Error calculating MFI: {str(e)}")
            raise
    
    def calculate_obv(self, df):
        """Calculate On-Balance Volume using vectorized operations."""
        try:
            # Initialize OBV
            close_change = df['close'].diff()
            volume = df['volume'].values
            
            # Use numpy where for vectorized calculation
            volume_adj = np.where(close_change > 0, volume, 
                                np.where(close_change < 0, -volume, 0))
            
            obv = pd.Series(volume_adj, index=df.index).cumsum()
            return obv
        except Exception as e:
            print(f"Error calculating OBV: {str(e)}")
            raise
    
    def calculate_historical_volatility(self, df, window=20):
        """Calculate historical volatility using vectorized operations."""
        try:
            returns = df['close'].pct_change()
            vol = returns.rolling(window).std() * np.sqrt(252)
            return vol
        except Exception as e:
            print(f"Error calculating volatility: {str(e)}")
            raise
    
    def calculate_relative_volatility(self, df, window=20):
        """Calculate relative volatility using vectorized operations."""
        try:
            hist_vol = self.calculate_historical_volatility(df, window)
            vol_ma = hist_vol.rolling(window=window*5).mean()  # Longer-term average
            rel_vol = hist_vol / vol_ma
            return rel_vol
        except Exception as e:
            print(f"Error calculating relative volatility: {str(e)}")
            raise
    
    def determine_regime(self, df):
        """Determine market regime using vectorized operations."""
        try:
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
            
            # Calculate regime strength
            regime_strength = pd.Series(0.0, index=df.index)
            regime_strength[bull_mask] = vol[bull_mask] / vol_threshold[bull_mask]
            regime_strength[bear_mask] = -vol[bear_mask] / vol_threshold[bear_mask]
            
            return regime, regime_strength
        except Exception as e:
            print(f"Error determining regime: {str(e)}")
            raise
    
    def select_strike_price(self, current_price, volatility, rel_vol):
        """Select strike price based on volatility and regime."""
        try:
            # Calculate ATM strike
            strike = current_price
            
            # Adjust based on volatility
            if rel_vol > 1.1:  # High volatility
                # Go slightly OTM
                strike *= 1.02
            elif rel_vol < 0.9:  # Low volatility
                # Go slightly ITM
                strike *= 0.98
                
            return strike
            
        except Exception as e:
            print(f"Error selecting strike: {str(e)}")
            raise
    
    def calculate_position_size(self, account_size, volatility, regime_strength):
        """Calculate position size based on volatility and regime strength."""
        try:
            # Base size using account risk
            risk_dollars = account_size * self.max_risk_pct
            
            # Estimate option price (rough approximation)
            est_option_price = 2.50  # Average price for slightly OTM options
            
            # Calculate number of contracts
            max_contracts = risk_dollars / (est_option_price * 100)  # 100 shares per contract
            
            # Adjust for volatility - reduce size in high vol
            vol_adj = 1 / (1 + (volatility - 0.3))  # Normalize around 30% vol
            contracts = max_contracts * vol_adj
            
            # Adjust for regime strength
            if regime_strength > 0:  # Bullish
                regime_mult = 1 + (regime_strength * 0.2)  # Max 20% increase
            else:  # Bearish
                regime_mult = 1 - (abs(regime_strength) * 0.5)  # Max 50% decrease
                
            contracts *= regime_mult
            
            # Round down to nearest contract
            return max(1, int(contracts))  # Minimum 1 contract
            
        except Exception as e:
            print(f"Error calculating position size: {str(e)}")
            raise
