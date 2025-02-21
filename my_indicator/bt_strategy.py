import backtrader as bt
import numpy as np
from typing import Dict

class OBVIndicator(bt.Indicator):
    """On Balance Volume with rolling window"""
    lines = ('obv',)
    params = (
        ('period', None),
        ('ema_period', None)
    )
    plotinfo = dict(subplot=True)
    plotlines = dict(obv=dict(_name='OBV', color='green'))
    
    def __init__(self):
        # Calculate price change direction
        price_change = self.data.close - self.data.close(-1)
        
        # Calculate volume with direction
        volume = bt.If(
            price_change > 0,
            self.data.volume,
            bt.If(
                price_change < 0,
                -self.data.volume,
                0.0
            )
        )
        
        # Use EMA for rolling OBV calculation with zero handling
        raw_obv = bt.indicators.SumN(volume, period=self.p.period)
        self.lines.obv = bt.If(
            raw_obv != 0,
            bt.indicators.EMA(raw_obv, period=self.p.ema_period),
            0.0
        )

class MFIIndicator(bt.Indicator):
    """Money Flow Index - Volume-weighted RSI"""
    lines = ('mfi',)
    params = (('period', None),)
    plotinfo = dict(subplot=True)
    
    def __init__(self):
        # Ensure valid prices
        valid_high = bt.Max(self.data.high, 0.00001)
        valid_low = bt.Max(self.data.low, 0.00001)
        valid_close = bt.Max(self.data.close, 0.00001)
        
        # Calculate typical price with minimum value
        typical_price = bt.Max((valid_high + valid_low + valid_close) / 3.0, 0.00001)
        
        # Calculate raw money flow with minimum volume
        valid_volume = bt.Max(self.data.volume, 0.00001)
        money_flow = typical_price * valid_volume
        
        # Calculate direction with smoothing to reduce noise
        direction = typical_price - typical_price(-1)
        smoothed_direction = bt.indicators.EMA(direction, period=min(3, self.p.period))
        
        # Split flows with minimum values
        pos_flow = bt.If(smoothed_direction > 0, money_flow, 0.00001)
        neg_flow = bt.If(smoothed_direction < 0, money_flow, 0.00001)
        
        # Sum flows with minimum values
        pos_sum = bt.Max(bt.indicators.SumN(pos_flow, period=self.p.period), 0.00001)
        neg_sum = bt.Max(bt.indicators.SumN(neg_flow, period=self.p.period), 0.00001)
        
        # Calculate ratio with bounds
        raw_ratio = pos_sum / neg_sum
        bounded_ratio = bt.Max(0.0, bt.Min(100.0, raw_ratio))
        
        # Calculate MFI with smoothing
        raw_mfi = 100.0 - (100.0 / (1.0 + bounded_ratio))
        smoothed_mfi = bt.indicators.EMA(raw_mfi, period=min(3, self.p.period))
        
        # Ensure final MFI stays within valid range
        self.lines.mfi = bt.Max(0.0, bt.Min(100.0, smoothed_mfi))

class RegimeIndicator(bt.Indicator):
    """Indicator that determines market regime based on trend and volatility"""
    lines = ('regime', 'trend', 'current_vol', 'vol_threshold')
    params = (
        ('ema_short_span', None),
        ('ema_long_span', None),
        ('min_trend_strength', None),
        ('current_window', None),
        ('historical_window', None),
        ('volatility_multiplier', None),
        ('volatility_cap', None)
    )
    plotinfo = dict(subplot=True)
    
    def __init__(self):
        # Price EMAs for trend
        self.ema_short = bt.indicators.EMA(self.data.close, period=self.p.ema_short_span)
        self.ema_long = bt.indicators.EMA(self.data.close, period=self.p.ema_long_span)
        
        # True Range for volatility with more robust zero handling
        self.tr = bt.indicators.TrueRange()
        
        # Ensure close price is valid for percentage calculation
        valid_close = bt.Max(self.data.close, 0.00001)  # Minimum valid price
        
        # Calculate TR percentage with bounds
        raw_tr_pct = 100.0 * self.tr / valid_close
        self.tr_pct = bt.Max(0.0, bt.Min(100.0, raw_tr_pct))  # Bound between 0-100%
        
        # Current volatility with smoothing
        self.lines.current_vol = bt.indicators.EMA(
            bt.indicators.EMA(self.tr_pct, period=min(20, self.p.current_window)),  # Short-term smoothing
            period=self.p.current_window
        )
        
        # Historical volatility with double smoothing and caps
        smoothed_tr = bt.indicators.EMA(
            self.tr_pct,
            period=min(self.p.historical_window // 4, self.p.volatility_cap // 4)
        )
        
        historical_vol = bt.indicators.EMA(
            smoothed_tr,
            period=min(self.p.historical_window, self.p.volatility_cap)
        )
        
        historical_std = bt.indicators.StdDev(
            smoothed_tr,
            period=min(self.p.historical_window, self.p.volatility_cap)
        )
        
        # Trend calculation with robust zero handling and smoothing
        trend_diff = self.ema_short - self.ema_long
        valid_price = bt.Max(self.data.close, 0.00001)  # Minimum valid price
        
        # Calculate normalized trend with bounds
        raw_trend = trend_diff / valid_price
        bounded_trend = bt.Max(-0.1, bt.Min(0.1, raw_trend))  # Limit trend strength
        
        # Apply additional smoothing
        self.lines.trend = bt.indicators.EMA(bounded_trend, period=5)
        
        # Volatility threshold with minimum floor
        vol_threshold = historical_vol + (historical_std * self.p.volatility_multiplier)
        self.lines.vol_threshold = bt.Max(0.001, vol_threshold)  # Ensure minimum threshold
        
    def next(self):
        # Default to no regime (0)
        self.lines.regime[0] = 0
        
        # Check if volatility condition is met
        if self.lines.current_vol[0] > self.lines.vol_threshold[0]:
            # Check trend direction and strength
            if self.lines.trend[0] > self.p.min_trend_strength:
                self.lines.regime[0] = 1  # bull_high_vol
            elif self.lines.trend[0] < -self.p.min_trend_strength:
                self.lines.regime[0] = -1  # bear_high_vol

class MinuteConfirmation(bt.Indicator):
    """Indicator for minute-by-minute confirmation of entry signals"""
    lines = ('confirmed', 'price_vol', 'vol_change', 'spread_pct')
    params = (
        ('min_vol_increases', None),
        ('max_spread_pct', None),
        ('max_price_volatility', None),
        ('period', None),
        ('vol_trend_threshold', None),
        ('vol_threshold_multiplier', None)
    )
    plotinfo = dict(subplot=True)
    
    def __init__(self):
        # Price volatility with rolling window
        price_changes = bt.indicators.PercentChange(self.data.close, period=1)
        self.lines.price_vol = bt.indicators.EMA(
            bt.indicators.StdDev(price_changes, period=self.p.period),
            period=self.p.period
        )
        
        # Volume change with rolling window
        raw_vol_change = bt.indicators.Momentum(self.data.volume, period=1)
        self.lines.vol_change = bt.indicators.EMA(raw_vol_change, period=self.p.period)
        
        # Spread calculation with robust zero handling
        spread = self.data.high - self.data.low
        self.lines.spread_pct = bt.If(
            self.data.low > 0,
            spread / self.data.low,
            bt.If(
                spread > 0,
                1.0,    # Maximum spread if low is 0 but there is a spread
                0.0     # No spread if both high and low are 0
            )
        )
        
    def next(self):
        # Default to not confirmed
        self.lines.confirmed[0] = 0
        
        if len(self) >= self.p.period:
            # Check volume momentum
            vol_increases = sum(1 for i in range(self.p.period) if self.lines.vol_change[-i] > 0)
            volume_trend = sum(self.lines.vol_change.get(size=self.p.period))
            
            # Check spread stability
            spread_stable = self.lines.spread_pct[0] < self.p.max_spread_pct
            
            # Use configured thresholds
            if (volume_trend > self.p.vol_trend_threshold and
                vol_increases >= self.p.min_vol_increases and
                self.lines.price_vol[0] < self.p.max_price_volatility * self.p.vol_threshold_multiplier and
                spread_stable):
                self.lines.confirmed[0] = 1

class VAMEStrategy(bt.Strategy):
    # Add plot lines for signals
    lines = ('long_signal', 'short_signal', 'exit_signal')
    plotinfo = dict(plot=True)
    plotlines = dict(
        long_signal=dict(_name='Long', color='lime', marker='^', markersize=8.0, ls=''),
        short_signal=dict(_name='Short', color='red', marker='v', markersize=8.0, ls=''),
        exit_signal=dict(_name='Exit', color='black', marker='x', markersize=8.0, ls='')
    )
    
    params = (
        # Output control
        ('verbose', True),  # Control logging output
        
        # Core parameters
        ('mfi_period', None),
        ('vwap_window', None),
        ('atr_period', None),
        ('min_stop_pct', None),
        ('max_stop_pct', None),
        ('risk_per_trade', None),
        ('vwap_exit_buffer_bull', None),
        ('vwap_exit_buffer_bear', None),
        ('min_hold_bars', None),
        ('ema_short_span', None),
        ('ema_long_span', None),
        ('min_trend_strength', None),
        ('current_window', None),
        ('historical_window', None),
        ('volatility_multiplier', None),
        ('mfi_bull_entry', None),
        ('mfi_bear_entry', None),
        ('mfi_bull_exit', None),
        ('mfi_bear_exit', None),
        ('allow_shorts', None),
        ('profit_target_pct', None),
        ('profit_lock_pct', None),
        
        # Indicator parameters
        ('obv_period', 20),
        ('obv_ema_period', 20),
        ('vol_ema_period', 14),
        ('price_vol_period', 5),
        ('min_vol_increases', 1),
        ('max_spread_pct', 0.006),
        ('max_price_volatility', 0.003),
        ('volatility_cap', 250),
        ('vol_trend_threshold', -0.2),
        ('vol_threshold_multiplier', 1.5)
    )
    
    def __init__(self):
        # Add cooldown tracking
        self.last_trade_bar = 0
        self.min_bars_between_trades = 5  # Default minimum bars between trades
        self.stop_loss_cooldown = 15  # Longer cooldown after stop loss
        self.last_trade_was_loss = False  # Track if last trade was stopped out
        self.min_profit_target = 0.002  # 0.2% minimum profit before trailing stop activates
        
        # Core indicators with robust zero handling
        # Ensure valid prices for VWAP
        valid_high = bt.Max(self.data.high, 0.00001)
        valid_low = bt.Max(self.data.low, 0.00001)
        valid_close = bt.Max(self.data.close, 0.00001)
        
        # Calculate VWAP with smoothing
        typical_price = (valid_high + valid_low + valid_close) / 3.0
        smoothed_price = bt.indicators.EMA(typical_price, period=min(5, self.p.vwap_window))
        self.vwap = bt.indicators.WeightedMovingAverage(
            smoothed_price,
            period=self.p.vwap_window,
            subplot=False
        )
        
        # OBV calculation
        self.obv = OBVIndicator(
            period=self.p.obv_period,
            ema_period=self.p.obv_ema_period
        )
        self.obv_diff = self.obv - self.obv(-1)
        self.mfi = MFIIndicator(period=self.p.mfi_period)
        self.atr = bt.indicators.ATR(period=self.p.atr_period, subplot=True)
        
        # Regime classification
        self.regime = RegimeIndicator(
            ema_short_span=self.p.ema_short_span,
            ema_long_span=self.p.ema_long_span,
            min_trend_strength=self.p.min_trend_strength,
            current_window=self.p.current_window,
            historical_window=self.p.historical_window,
            volatility_multiplier=self.p.volatility_multiplier,
            volatility_cap=self.p.volatility_cap
        )
        
        # Entry confirmation
        self.minute_confirmed = MinuteConfirmation(
            min_vol_increases=self.p.min_vol_increases,
            max_spread_pct=self.p.max_spread_pct,
            max_price_volatility=self.p.max_price_volatility,
            period=self.p.price_vol_period,
            vol_trend_threshold=self.p.vol_trend_threshold,
            vol_threshold_multiplier=self.p.vol_threshold_multiplier
        )
        
        # Position tracking
        self.entry_price = None
        self.entry_bar = None
        self.highest_price = None
        self.lowest_price = None
        
    def log_indicators(self):
        """Log current indicator values"""
        return (
            f"\nIndicator Values:"
            f"\n  Close: {self.data.close[0]:.2f}"
            f"\n  VWAP: {self.vwap[0]:.2f}"
            f"\n  OBV Diff: {self.obv_diff[0]:.2f}"
            f"\n  MFI: {self.mfi[0]:.2f}"
            f"\n  Regime: {self.regime.regime[0]}"
            f"\n  Trend: {self.regime.trend[0]:.4f}"
            f"\n  Current Vol: {self.regime.current_vol[0]:.4f}"
            f"\n  Vol Threshold: {self.regime.vol_threshold[0]:.4f}"
            f"\n  Minute Confirmed: {self.minute_confirmed.confirmed[0]}"
        )

    def notify_trade(self, trade):
        if trade.isclosed:
            bars_held = len(self) - self.entry_bar if self.entry_bar is not None else 0
            pnl = trade.pnl
            
            # Check if this was a loss and if it was stopped out
            was_stopped = False
            if pnl < 0:
                # For longs, check if we hit the trailing stop
                if self.position.size > 0 and self.highest_price is not None:  # Was long
                    current_profit_pct = (self.data.close[0] / self.entry_price) - 1
                    if current_profit_pct >= self.min_profit_target:
                        stop_price = self.highest_price * (1 - self.p.profit_lock_pct)
                        was_stopped = self.data.close[0] <= stop_price
                        if was_stopped:
                            if self.p.verbose:
                                print(f'Long stop loss triggered at {self.data.close[0]:.2f} (stop: {stop_price:.2f})')
                # For shorts, check if we hit the trailing stop
                elif self.position.size < 0 and self.lowest_price is not None:  # Was short
                    current_profit_pct = 1 - (self.data.close[0] / self.entry_price)
                    if current_profit_pct >= self.min_profit_target:
                        stop_price = self.lowest_price * (1 + self.p.profit_lock_pct)
                        was_stopped = self.data.close[0] >= stop_price
                        if was_stopped:
                            if self.p.verbose:
                                print(f'Short stop loss triggered at {self.data.close[0]:.2f} (stop: {stop_price:.2f})')
                
                if was_stopped:
                    self.last_trade_was_loss = True
                    if self.p.verbose:
                        print(f'Setting longer cooldown period: {self.stop_loss_cooldown} bars')
            
            if self.p.verbose:
                print(f'\nTrade Closed - PnL: ${pnl:.2f}'
                  f'\n  Entry Price: ${trade.price:.2f}'
                  f'\n  Exit Price: ${trade.data.close[0]:.2f}'
                  f'\n  Bars Held: {bars_held}'
                  f'\n  Was Stopped: {was_stopped}'
                  f'{self.log_indicators()}')
            
            # Update last trade bar on close
            self.last_trade_bar = len(self)
    
    def get_position_size(self, price, atr):
        """Calculate position size based on risk"""
        stop_pct = min(max(atr / price, self.p.min_stop_pct), self.p.max_stop_pct)
        risk_amount = self.broker.getcash() * self.p.risk_per_trade
        size = risk_amount / (price * stop_pct)
        return int(size)
    
    def can_trade(self):
        """Check if enough bars have passed since last trade"""
        bars_since_last = len(self) - self.last_trade_bar
        required_bars = self.stop_loss_cooldown if self.last_trade_was_loss else self.min_bars_between_trades
        
        if bars_since_last >= required_bars:
            self.last_trade_was_loss = False  # Reset loss flag when cooldown expires
            return True
        return False
    
    def should_long(self):
        """Check if we should enter a long position"""
        return (
            self.regime.regime[0] == 1 and  # bull_high_vol
            self.data.close[0] > self.vwap[0] and  # price above VWAP
            self.data.close[0] > self.data.close[-1] and  # immediate uptrend
            self.obv_diff[0] > 0 and  # current OBV rising
            self.obv_diff[-1] > 0 and  # previous OBV also rising
            self.mfi[0] > self.p.mfi_bull_entry and  # MFI bullish
            self.mfi[0] < 90 and  # not overbought
            self.minute_confirmed.confirmed[0] == 1  # minute confirmation
        )
    
    def should_short(self):
        """Check if we should enter a short position"""
        return (
            self.p.allow_shorts and
            self.regime.regime[0] == -1 and  # bear_high_vol
            self.data.close[0] < self.vwap[0] and  # price below VWAP
            self.data.close[0] < self.data.close[-1] and  # immediate downtrend
            self.obv_diff[0] < 0 and  # current OBV falling
            self.obv_diff[-1] < 0 and  # previous OBV also falling
            self.mfi[0] < self.p.mfi_bear_entry and  # MFI bearish
            self.mfi[0] > 10 and  # not oversold
            self.minute_confirmed.confirmed[0] == 1  # minute confirmation
        )
    
    def should_exit_long(self):
        """Check if we should exit a long position"""
        # Must hold for minimum number of bars
        if len(self) - self.entry_bar < self.p.min_hold_bars:
            return False
            
        # Update trailing stop
        if self.highest_price is not None:
            self.highest_price = max(self.highest_price, self.data.close[0])
            current_profit_pct = (self.data.close[0] / self.entry_price) - 1
            
            # Only activate trailing stop after minimum profit target
            if current_profit_pct >= self.min_profit_target:
                profit_range_pct = (self.entry_price * (1 + self.p.profit_target_pct)) / self.entry_price - 1
                new_stop = self.highest_price * (1 - (profit_range_pct * self.p.profit_lock_pct))
                if self.data.close[0] <= new_stop:
                    return True
        
        # VWAP exit
        buffer_factor = 1 - self.p.vwap_exit_buffer_bull
        if self.data.close[0] < (self.vwap[0] * buffer_factor):
            return True
            
        # MFI exit
        return self.mfi[0] > self.p.mfi_bull_exit
    
    def should_exit_short(self):
        """Check if we should exit a short position"""
        # Must hold for minimum number of bars
        if len(self) - self.entry_bar < self.p.min_hold_bars:
            return False
            
        # Update trailing stop
        if self.lowest_price is not None:
            self.lowest_price = min(self.lowest_price, self.data.close[0])
            current_profit_pct = 1 - (self.data.close[0] / self.entry_price)
            
            # Only activate trailing stop after minimum profit target
            if current_profit_pct >= self.min_profit_target:
                profit_range_pct = 1 - (self.entry_price * (1 - self.p.profit_target_pct)) / self.entry_price
                new_stop = self.lowest_price * (1 + (profit_range_pct * self.p.profit_lock_pct))
                if self.data.close[0] >= new_stop:
                    return True
        
        # VWAP exit
        buffer_factor = 1 + self.p.vwap_exit_buffer_bear
        if self.data.close[0] > (self.vwap[0] * buffer_factor):
            return True
            
        # MFI exit
        return self.mfi[0] < self.p.mfi_bear_exit
    
    def next(self):
        # Initialize signal lines
        self.lines.long_signal[0] = float('nan')
        self.lines.short_signal[0] = float('nan')
        self.lines.exit_signal[0] = float('nan')
        
        # If we're not in the market and cooldown period has passed
        if not self.position and self.can_trade():
            # Check long conditions
            if self.should_long():
                size = self.get_position_size(self.data.close[0], self.atr[0])
                if self.p.verbose:
                    print(f'\nLong Entry Signal at {self.data.datetime.datetime()}:'
                      f'\n  Price: ${self.data.close[0]:.2f}'
                      f'\n  Size: {size}'
                      f'\n  Conditions:'
                      f'\n    Regime: {self.regime.regime[0]}'
                      f'\n    Close vs VWAP: {self.data.close[0]:.2f} vs {self.vwap[0]:.2f}'
                      f'\n    OBV Diff: {self.obv_diff[0]:.2f}'
                      f'\n    MFI: {self.mfi[0]:.2f} vs {self.p.mfi_bull_entry}'
                      f'{self.log_indicators()}')
                # Plot long entry signal
                self.lines.long_signal[0] = self.data.close[0]
                self.buy(size=size, exectype=bt.Order.Market)
                self.entry_price = self.data.close[0]
                self.entry_bar = len(self)
                self.highest_price = self.data.close[0]
            
            # Check short conditions
            elif self.should_short():
                size = self.get_position_size(self.data.close[0], self.atr[0])
                if self.p.verbose:
                    print(f'\nShort Entry Signal at {self.data.datetime.datetime()}:'
                      f'\n  Price: ${self.data.close[0]:.2f}'
                      f'\n  Size: {size}'
                      f'\n  Conditions:'
                      f'\n    Regime: {self.regime.regime[0]}'
                      f'\n    Close vs VWAP: {self.data.close[0]:.2f} vs {self.vwap[0]:.2f}'
                      f'\n    OBV Diff: {self.obv_diff[0]:.2f}'
                      f'\n    MFI: {self.mfi[0]:.2f} vs {self.p.mfi_bear_entry}'
                      f'{self.log_indicators()}')
                # Plot short entry signal
                self.lines.short_signal[0] = self.data.close[0]
                self.sell(size=size, exectype=bt.Order.Market)
                self.entry_price = self.data.close[0]
                self.entry_bar = len(self)
                self.lowest_price = self.data.close[0]
                
        # If we're in a long position
        elif self.position.size > 0:
            # First check if minimum hold period has passed
            bars_held = len(self) - self.entry_bar
            if bars_held < self.p.min_hold_bars:
                return
            
            # Update trailing stop
            trailing_stop = False
            if self.highest_price is not None:
                self.highest_price = max(self.highest_price, self.data.close[0])
                current_profit_pct = (self.data.close[0] / self.entry_price) - 1
                
                # Only activate trailing stop after minimum profit target
                if current_profit_pct >= self.min_profit_target:
                    profit_range_pct = (self.entry_price * (1 + self.p.profit_target_pct)) / self.entry_price - 1
                    new_stop = self.highest_price * (1 - (profit_range_pct * self.p.profit_lock_pct))
                    trailing_stop = self.data.close[0] <= new_stop
            
            # Check VWAP exit
            buffer_factor = 1 - self.p.vwap_exit_buffer_bull
            vwap_exit = self.data.close[0] < (self.vwap[0] * buffer_factor)
            
            # Check MFI exit
            mfi_exit = self.mfi[0] > self.p.mfi_bull_exit
            
            # Exit if any condition is met after minimum hold period
            if trailing_stop or vwap_exit or mfi_exit:
                # Calculate current P&L
                current_profit_pct = (self.data.close[0] / self.entry_price) - 1
                pnl = (self.data.close[0] - self.entry_price) * self.position.size
                
                # Build detailed exit reason
                exit_reasons = []
                if trailing_stop:
                    exit_reasons.append(f'Trailing Stop: Hit stop at ${self.data.close[0]:.2f} (stop: ${new_stop:.2f})')
                if vwap_exit:
                    exit_reasons.append(f'VWAP Exit: Price ${self.data.close[0]:.2f} < VWAP ${self.vwap[0]:.2f} * {1-buffer_factor:.3f}')
                if mfi_exit:
                    exit_reasons.append(f'MFI Exit: {self.mfi[0]:.1f} > {self.p.mfi_bull_exit} (exit threshold)')
                
                if self.p.verbose:
                    print(f'\nLong Exit Signal at {self.data.datetime.datetime()}:'
                      f'\n  Price: ${self.data.close[0]:.2f}'
                      f'\n  Entry: ${self.entry_price:.2f}'
                      f'\n  P&L: ${pnl:.2f} ({current_profit_pct:.2%})'
                      f'\n  Bars Held: {bars_held}'
                      f'\n  Exit Reasons:'
                      f'\n    ' + '\n    '.join(exit_reasons) +
                      f'{self.log_indicators()}')
                # Plot exit signal
                self.lines.exit_signal[0] = self.data.close[0]
                self.close(exectype=bt.Order.Market)  # Force immediate execution
                self.highest_price = None
                self.entry_price = None
                self.entry_bar = None
                
        # If we're in a short position
        elif self.position.size < 0:
            # First check if minimum hold period has passed
            bars_held = len(self) - self.entry_bar
            if bars_held < self.p.min_hold_bars:
                return
            
            # Update trailing stop
            trailing_stop = False
            if self.lowest_price is not None:
                self.lowest_price = min(self.lowest_price, self.data.close[0])
                current_profit_pct = 1 - (self.data.close[0] / self.entry_price)
                
                # Only activate trailing stop after minimum profit target
                if current_profit_pct >= self.min_profit_target:
                    profit_range_pct = 1 - (self.entry_price * (1 - self.p.profit_target_pct)) / self.entry_price
                    new_stop = self.lowest_price * (1 + (profit_range_pct * self.p.profit_lock_pct))
                    trailing_stop = self.data.close[0] >= new_stop
            
            # Check VWAP exit
            buffer_factor = 1 + self.p.vwap_exit_buffer_bear
            vwap_exit = self.data.close[0] > (self.vwap[0] * buffer_factor)
            
            # Check MFI exit
            mfi_exit = self.mfi[0] < self.p.mfi_bear_exit
            
            # Exit if any condition is met after minimum hold period
            if trailing_stop or vwap_exit or mfi_exit:
                # Calculate current P&L
                current_profit_pct = 1 - (self.data.close[0] / self.entry_price)
                pnl = (self.entry_price - self.data.close[0]) * abs(self.position.size)
                
                # Build detailed exit reason
                exit_reasons = []
                if trailing_stop:
                    exit_reasons.append(f'Trailing Stop: Hit stop at ${self.data.close[0]:.2f} (stop: ${new_stop:.2f})')
                if vwap_exit:
                    exit_reasons.append(f'VWAP Exit: Price ${self.data.close[0]:.2f} > VWAP ${self.vwap[0]:.2f} * {1+buffer_factor:.3f}')
                if mfi_exit:
                    exit_reasons.append(f'MFI Exit: {self.mfi[0]:.1f} < {self.p.mfi_bear_exit} (exit threshold)')
                
                if self.p.verbose:
                    print(f'\nShort Exit Signal at {self.data.datetime.datetime()}:'
                      f'\n  Price: ${self.data.close[0]:.2f}'
                      f'\n  Entry: ${self.entry_price:.2f}'
                      f'\n  P&L: ${pnl:.2f} ({current_profit_pct:.2%})'
                      f'\n  Bars Held: {bars_held}'
                      f'\n  Exit Reasons:'
                      f'\n    ' + '\n    '.join(exit_reasons) +
                      f'{self.log_indicators()}')
                # Plot exit signal
                self.lines.exit_signal[0] = self.data.close[0]
                self.close(exectype=bt.Order.Market)  # Force immediate execution
                self.lowest_price = None
                self.entry_price = None
                self.entry_bar = None
