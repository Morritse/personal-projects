##############################################################################
# vwap_obv_strategy.py
##############################################################################
import pandas as pd
from typing import Dict, List
from datetime import datetime
import talib
from utils import Indicators, RegimeClassifier, PositionSizer, MinuteConfirmation, SignalGenerator

class VAMEStrategy:
    def __init__(self, config: Dict):
        self.config = config
        
        self.indicators          = Indicators()
        self.regime_classifier   = RegimeClassifier(config)
        self.position_sizer      = PositionSizer(config)
        self.minute_confirmation = MinuteConfirmation(config)
        self.signal_generator    = SignalGenerator(config)
        
        # Extract single-element lists as plain ints/floats
        self.mfi_period   = config['MFI Period'][0]
        self.vwap_window  = config['VWAP Window'][0]
        self.atr_period   = config['ATR Period'][0]
        
        self.min_stop_pct = config['Min Stop Pct'][0]
        self.max_stop_pct = config['Max Stop Pct'][0]
        self.risk_per_trade = config['Risk Per Trade'][0]
        
        self.regime_params = config['regime_params']
        
        ### NEW: user-configurable buffer for VWAP exit 
        self.vwap_exit_buffer_bull = 0.005  # 0.5% below VWAP
        self.vwap_exit_buffer_bear = 0.005  # 0.5% above VWAP
        
        ### NEW: how many bars to hold before allowing an exit
        self.min_hold_bars = 3  # you can change to 2,3,5, etc.

    def precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['vwap'] = self.indicators.calculate_vwap(
            df['high'].values, 
            df['low'].values,
            df['close'].values,
            df['volume'].values,
            self.vwap_window
        )
        df['obv']      = talib.OBV(df['close'], df['volume'])
        df['obv_diff'] = df['obv'].diff()
        df['mfi']      = talib.MFI(
            df['high'], df['low'],
            df['close'], df['volume'],
            timeperiod=self.mfi_period
        )
        df['atr']      = talib.ATR(
            df['high'], df['low'],
            df['close'], timeperiod=self.atr_period
        )
        return df
    
    def get_regime_parameters(self, regime: str) -> Dict:
        return self.regime_params.get(regime, {})
    
    def run(self, df_1m: pd.DataFrame) -> List[Dict]:
        """
        Entire strategy on 1-minute data. 
        Return *only* the trades that occur on the last bar 
        so that we don't repeatedly process old trades.
        """
        trades = []
        try:
            df = self.precompute_indicators(df_1m)
            
            # Warmup
            warmup_bars = max(self.mfi_period, self.vwap_window, self.atr_period)
            if len(df) < warmup_bars:
                return []
            
            df = df.iloc[warmup_bars:].copy()
            
            # Classify regime
            df['regime'] = self.regime_classifier.classify(df)
            # Generate signals
            df = self.signal_generator.generate_signals(df)
            
            # We'll track position in a pseudo way
            position = None
            bar_entered = None  # track the index (i) at which we entered
            highest_price = None
            
            for i in range(len(df)):
                bar = df.iloc[i]
                
                if not position:
                    # Potential entry
                    if (bar['regime']
                      and bar['price_signal']
                      and bar['obv_signal']
                      and bar['mfi_signal']):
                        
                        # Check minute confirmation
                        if not self.minute_confirmation.check_confirmation(df_1m, bar.name, bar['regime']):
                            continue
                        
                        # Grab regime params
                        params = self.get_regime_parameters(bar['regime'])
                        if not params:
                            continue
                        
                        # Sizing
                        size = self.position_sizer.calculate_size(
                            bar['close'],
                            bar['atr'],
                            25_000,   # e.g. your capital
                            params,
                            1.0
                        )
                        
                        # Stop + target
                        raw_stop_pct = ((params['stop_mult'][0] if isinstance(params['stop_mult'], list)
                                         else params['stop_mult'])
                                        * bar['atr']) / bar['close']
                        stop_pct = min(
                            max(raw_stop_pct, self.min_stop_pct),
                            self.max_stop_pct
                        )
                        reward_pct = stop_pct * (params['reward_risk'][0] if isinstance(params['reward_risk'], list)
                                                 else params['reward_risk'])
                        
                        if bar['regime'] == 'bull_high_vol':
                            stop_loss   = bar['close']*(1 - stop_pct)
                            take_profit = bar['close']*(1 + reward_pct)

                        elif bar['regime'] == 'bear_high_vol':
                            # DISABLE short trades:
                            if not self.config.get('allow_shorts', True):
                                # skip creating a short trade altogether
                                continue                       
                            else:
                                stop_loss   = bar['close']*(1 + stop_pct)
                                take_profit = bar['close']*(1 - reward_pct)
                        
                        position = {
                            'entry_time':  bar.name,
                            'entry_price': bar['close'],
                            'stop_loss':   stop_loss,
                            'take_profit': take_profit,
                            'size':        size,
                            'regime':      bar['regime']
                        }
                        bar_entered = i
                        
                        # For trailing stops 
                        # (only if you want that; we keep it from your original)
                        trailing_on = (params['trailing_stop'][0] 
                                       if isinstance(params['trailing_stop'], list) 
                                       else params['trailing_stop'])
                        highest_price = bar['close'] if trailing_on else None
                        
                        trades.append({
                            'timestamp':   bar.name,
                            'action':      'BUY' if bar['regime']=='bull_high_vol' else 'SELL',
                            'price':       bar['close'],
                            'size':        size,
                            'stop_loss':   stop_loss,
                            'take_profit': take_profit,
                            'regime':      bar['regime']
                        })
                
                else:
                    # Already in a position => check exit
                    hold_bars = i - bar_entered
                    if position['regime'] == 'bull_high_vol':
                        # Trailing
                        if highest_price is not None:
                            highest_price = max(highest_price, bar['close'])
                            profit_range_pct = (position['take_profit']/position['entry_price']) - 1
                            new_stop = highest_price * (1 - (profit_range_pct * 0.5))
                            position['stop_loss'] = max(position['stop_loss'], new_stop)
                        
                        hit_stop   = (bar['close'] <= position['stop_loss'])
                        hit_target = (bar['close'] >= position['take_profit'])
                        
                        ### CHANGED: use a buffer so we only exit if close 
                        ### is below vwap by at least vwap_exit_buffer_bull
                        # Example: close < vwap * (1 - 0.5%)
                        buffer_factor = (1 - self.vwap_exit_buffer_bull)
                        vwap_exit = (bar['close'] < (bar['vwap'] * buffer_factor))
                        
                        # MFI exit
                        mfi_exit = (bar['mfi'] > self.config['mfi_exit']['bull'][0])
                    
                    else:
                        # bear regime
                        if highest_price is not None:
                            highest_price = min(highest_price, bar['close'])
                            profit_range_pct = 1 - (position['take_profit']/position['entry_price'])
                            new_stop = highest_price * (1 + (profit_range_pct * 0.5))
                            position['stop_loss'] = min(position['stop_loss'], new_stop)
                        
                        hit_stop   = (bar['close'] >= position['stop_loss'])
                        hit_target = (bar['close'] <= position['take_profit'])
                        
                        ### CHANGED: for bear exit, 
                        ### only exit if close is *above* vwap by vwap_exit_buffer_bear
                        buffer_factor = (1 + self.vwap_exit_buffer_bear)
                        vwap_exit = (bar['close'] > (bar['vwap'] * buffer_factor))
                        
                        # MFI exit
                        mfi_exit = (bar['mfi'] < self.config['mfi_exit']['bear'][0])
                    
                    # Enforce minimum hold bars
                    if hold_bars < self.min_hold_bars:
                        # skip exit checks entirely if we haven't held for X bars
                        continue
                    
                    # Now see if any exit condition is triggered
                    if hit_stop or hit_target or vwap_exit or mfi_exit:
                        pnl = (bar['close'] - position['entry_price']) * position['size']
                        if position['regime'] == 'bear_high_vol':
                            pnl = -pnl
                        
                        reason = 'stop_or_target' if (hit_stop or hit_target) else 'technical'
                        
                        trades.append({
                            'timestamp': bar.name,
                            'action':    'SELL' if position['regime']=='bull_high_vol' else 'BUY',
                            'price':     bar['close'],
                            'size':      position['size'],
                            'pnl':       pnl,
                            'reason':    reason,
                            'regime':    position['regime']
                        })
                        position     = None
                        highest_price= None
                        bar_entered  = None
            
            # ### NEW: Return only trades from the *final bar* 
            # so we don't keep re-processing old trades in your live script.
            if not trades:
                return []
            
            last_ts = df.index[-1]
            # Filter for trades that happened on the final bar
            final_trades = [t for t in trades if t['timestamp'] == last_ts]
            return final_trades

        except Exception as e:
            print(f"Error in strategy execution: {e}")
            return []
