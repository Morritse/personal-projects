##############################################################################
# utils.py
##############################################################################

import pandas as pd
import numpy as np
import talib
from datetime import datetime
from typing import Dict

class Indicators:
    """Module for indicator calculations."""
    
    @staticmethod
    def calculate_vwap(high: np.ndarray, low: np.ndarray, 
                       close: np.ndarray, volume: np.ndarray,
                       window: int) -> np.ndarray:
        typical_price = (high + low + close) / 3
        price_vol = pd.Series(typical_price * volume).rolling(window=window, min_periods=1).sum()
        vol_sum   = pd.Series(volume).rolling(window=window, min_periods=1).sum()
        vwap = price_vol / vol_sum
        return vwap.values


class RegimeClassifier:
    def __init__(self, config: Dict):
        self.current_window    = config.get('Current Window',[125])[0]
        self.historical_window = config.get('Historical Window',[1000])[0]
        self.volatility_multiplier = config.get('Volatility Multiplier',[1.0])[0]
        
        trend_params = config['trend_params']
        self.ema_short_span     = trend_params['ema_short_span'][0]
        self.ema_long_span      = trend_params['ema_long_span'][0]
        self.min_trend_strength = trend_params['min_trend_strength'][0]

    def classify(self, data: pd.DataFrame, debug: bool = False) -> pd.Series:
        """Classify bull_high_vol / bear_high_vol / NaN, with optional debug printing."""
        if len(data) < max(self.current_window, self.historical_window):
            return pd.Series(index=data.index, dtype='object')
        
        # Prepare a temp DataFrame
        df = pd.DataFrame(index=data.index)
        df['hl'] = data['high'] - data['low']
        df['hc'] = (data['high'] - data['close'].shift(1)).abs()
        df['lc'] = (data['low'] - data['close'].shift(1)).abs()
        df['tr'] = df[['hl','hc','lc']].max(axis=1)
        df['tr_pct'] = (df['tr']/data['close']) * 100
        
        # current and historical volatility
        df['current_vol']    = df['tr_pct'].rolling(self.current_window,   min_periods=1).mean()
        df['historical_vol'] = df['tr_pct'].rolling(self.historical_window, min_periods=1).mean()
        df['historical_std'] = df['tr_pct'].rolling(self.historical_window, min_periods=1).std()
        
        vol_threshold = df['historical_vol'] + (df['historical_std'] * self.volatility_multiplier)
        
        short_ema = data['close'].ewm(span=self.ema_short_span, adjust=False).mean()
        long_ema  = data['close'].ewm(span=self.ema_long_span,  adjust=False).mean()
        df['trend'] = (short_ema - long_ema) / data['close']
        
        bull_mask = (df['trend'] >  self.min_trend_strength) & (df['current_vol'] > vol_threshold)
        bear_mask = (df['trend'] < -self.min_trend_strength) & (df['current_vol'] > vol_threshold)
        
        regimes = pd.Series(index=data.index, dtype='object')
        regimes.loc[bull_mask] = 'bull_high_vol'
        regimes.loc[bear_mask] = 'bear_high_vol'
        
        # OPTIONAL: Debug print
        if len(df) > 0:
            # For the *last* bar's index:
            i = -1
            idx = df.index[i]
            print(
                f"[REGIME DEBUG] {idx} | "
                f"current_vol={df['current_vol'].iloc[i]:.2f}, "
                f"vol_threshold={vol_threshold.iloc[i]:.2f}, "
                f"trend={df['trend'].iloc[i]:.4f}, "
                f"Regime={regimes.iloc[i]}"
            )
        
        return regimes


class PositionSizer:
    def __init__(self, config: Dict):
        self.risk_per_trade = config.get('Risk Per Trade',[0.03])[0]
        self.min_stop_pct   = config.get('Min Stop Pct',[0.01])[0]
        self.max_stop_pct   = config.get('Max Stop Pct',[0.04])[0]
    
    def calculate_size(self, price: float, atr: float, capital: float,
                       regime_params: Dict, vol_mult: float) -> int:
        stop_mult = regime_params['stop_mult'][0] if isinstance(regime_params['stop_mult'], list) else regime_params['stop_mult']
        pos_scale = regime_params['position_scale'][0] if isinstance(regime_params['position_scale'], list) else regime_params['position_scale']
        
        raw_stop_pct = (atr*stop_mult)/price
        stop_pct     = min(max(raw_stop_pct,self.min_stop_pct),self.max_stop_pct)
        
        risk_amount   = capital*self.risk_per_trade
        risk_per_unit = price*stop_pct
        base_size     = risk_amount/risk_per_unit
        position_size = round(base_size * pos_scale * vol_mult)
        return max(1,position_size)


class MinuteConfirmation:
    def __init__(self, config: Dict):
        mc = config['minute_confirmation']
        self.min_vol_increases = mc['min_volume_increases'][0]
        self.max_spread_pct    = mc['max_spread_pct'][0]
        self.max_price_vol     = mc['max_price_volatility'][0]
        
    def check_confirmation(self, minute_data: pd.DataFrame, entry_time: datetime, regime: str) -> bool:
        try:
            recent_data = minute_data[minute_data.index <= entry_time].tail(5)
            if len(recent_data) < 5:
                return False
            
            price_changes = recent_data['close'].pct_change()
            price_trend   = price_changes.sum()
            price_vol     = price_changes.std()
            vol_changes   = recent_data['volume'].pct_change()
            volume_trend  = vol_changes.sum()
            volume_consistency = (vol_changes > 0).sum() >= self.min_vol_increases
            
            spreads = (recent_data['high']-recent_data['low'])/recent_data['low']
            spread_stable = (spreads.mean()<self.max_spread_pct)
            
            base_conditions = (
                volume_trend>-0.1
                and volume_consistency
                and price_vol<self.max_price_vol
                and spread_stable
            )
            if regime=='bull_high_vol':
                return base_conditions and price_trend>=-0.001
            else:
                return base_conditions and price_trend<=0.001
        except:
            return False


class SignalGenerator:
    def __init__(self, config: Dict):
        me = config['mfi_entry']
        mx = config['mfi_exit']
        self.bull_mfi_entry = me['bull'][0]
        self.bear_mfi_entry = me['bear'][0]
        self.bull_mfi_exit  = mx['bull'][0]
        self.bear_mfi_exit  = mx['bear'][0]
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['price_signal'] = False
        df['obv_signal']   = False
        df['mfi_signal']   = False
        
        bull_mask = (df['regime']=='bull_high_vol')
        bear_mask = (df['regime']=='bear_high_vol')
        
        # Price signal
        df.loc[bull_mask & (df['close']>df['vwap']), 'price_signal'] = True
        df.loc[bear_mask & (df['close']<df['vwap']), 'price_signal'] = True
        
        # OBV signal
        df.loc[bull_mask & (df['obv_diff']>0), 'obv_signal'] = True
        df.loc[bear_mask & (df['obv_diff']<0), 'obv_signal'] = True
        
        # MFI signal
        df.loc[bull_mask & (df['mfi']> self.bull_mfi_entry),'mfi_signal'] = True
        df.loc[bear_mask & (df['mfi']< self.bear_mfi_entry),'mfi_signal'] = True
        
        return df
