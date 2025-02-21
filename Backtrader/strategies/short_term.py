import backtrader as bt
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

class SafeRSI(bt.indicators.RSI):
    """RSI indicator with error handling for zero values"""
    
    def __init__(self):
        super(SafeRSI, self).__init__()
        self.lines.rsi = bt.Max(0.0001, self.lines.rsi)  # Prevent zero values

class SafeStochastic(bt.indicators.Stochastic):
    """Stochastic indicator with error handling for zero values"""
    
    def __init__(self):
        super(SafeStochastic, self).__init__()
        self.lines.percK = bt.Max(0.0001, self.lines.percK)  # Prevent zero values
        self.lines.percD = bt.Max(0.0001, self.lines.percD)  # Prevent zero values

class ShortTermStrategy(bt.Strategy):
    """
    Short-Term Mean Reversion Strategy using 1-minute data
    Pure signal evaluation of RSI and Stochastics
    """
    
    params = (
        # Indicator Parameters
        ('rsi_period', 5),         # Faster RSI
        ('stoch_period', 5),       # Faster Stochastic
        ('stoch_period_d', 3),     
        ('stoch_period_k', 3),     
        ('rsi_overbought', 75),    
        ('rsi_oversold', 25),
        ('stoch_overbought', 85),  
        ('stoch_oversold', 15),
        ('holding_period', 5),     # Quick mean reversion
        ('morning_session_start', time(9, 45)),  # Skip first 15 min
        ('morning_session_end', time(11, 30)),
        ('afternoon_session_start', time(13, 30)),
        ('afternoon_session_end', time(15, 45)),  # Skip last 15 min
    )
    
    def __init__(self):
        """Initialize indicators and tracking variables."""
        # Debug counters
        self.total_bars = 0
        self.filtered_bars = 0
        self.signal_bars = 0
        self.session_bars = 0
        
        # Technical Indicators with safeguards
        self.rsi = SafeRSI(
            self.data,
            period=self.p.rsi_period,
            safediv=True,  # Use safe division
            plotname='RSI'
        )
        
        self.stoch = SafeStochastic(
            self.data,
            period=self.p.stoch_period,
            period_dfast=self.p.stoch_period_d,
            period_dslow=self.p.stoch_period_k,
            safediv=True,  # Use safe division
            plotname='Stochastic'
        )
        
        # Moving average for trend identification
        self.sma = bt.indicators.SMA(
            self.data,
            period=20,
            plotname='20 SMA'
        )
        
        # Track position and signals
        self.entry_bar = 0
        self.trade_history = []
        self.signal_history = []  # For calculating IC
        
        # Performance tracking by market regime
        self.regime_stats = {
            'high_vol': {'trades': [], 'returns': []},
            'low_vol': {'trades': [], 'returns': []},
            'trending': {'trades': [], 'returns': []},
            'ranging': {'trades': [], 'returns': []}
        }
    
    def get_rsi_signal(self):
        """Get RSI signal: -1 (overbought), 1 (oversold), or 0 (neutral)."""
        try:
            rsi_value = self.rsi[0]
            if not rsi_value or not (0 <= rsi_value <= 100):
                return 0
                
            if rsi_value > self.p.rsi_overbought:
                return -1
            elif rsi_value < self.p.rsi_oversold:
                return 1
            return 0
        except:
            return 0
        
    def get_stoch_signal(self):
        """Get Stochastic signal: -1 (overbought), 1 (oversold), or 0 (neutral)."""
        try:
            k_value = self.stoch.percK[0]
            if not k_value or not (0 <= k_value <= 100):
                return 0
                
            if k_value > self.p.stoch_overbought:
                return -1
            elif k_value < self.p.stoch_oversold:
                return 1
            return 0
        except:
            return 0
    
    def next(self):
        """Process next bar and manage positions."""
        # Get base signals
        rsi_signal = self.get_rsi_signal()
        stoch_signal = self.get_stoch_signal()
        current_signal = 0
        
        # Update diagnostics
        self.total_bars += 1
        
        # Check if we're in valid trading hours
        current_time = self.data.datetime.datetime(0).time()
        if not (
            (self.p.morning_session_start <= current_time <= self.p.morning_session_end) or
            (self.p.afternoon_session_start <= current_time <= self.p.afternoon_session_end)
        ):
            return
            
        self.session_bars += 1
            
        # Only generate signal when both indicators strongly agree
        if rsi_signal == stoch_signal and rsi_signal != 0:
            # Simple signal generation
            if rsi_signal > 0:  # Potential buy
                if self.rsi[0] < self.p.rsi_oversold and self.stoch.percK[0] < self.p.stoch_oversold:
                    current_signal = 1
                    self.signal_bars += 1
            else:  # Potential sell
                if self.rsi[0] > self.p.rsi_overbought and self.stoch.percK[0] > self.p.stoch_overbought:
                    current_signal = -1
                    self.signal_bars += 1
            
        # Store signal for later analysis
        self.signal_history.append({
            'datetime': self.data.datetime.datetime(0),
            'signal': current_signal,
            'close': self.data.close[0],
            'rsi': self.rsi[0],
            'stoch_k': self.stoch.percK[0]
        })
        
        # Check if we need to exit based on holding period
        if self.position and (len(self) - self.entry_bar) >= self.p.holding_period:
            self.close_position("Time Exit")
            return
            
        # Only take new positions if we're flat
        if not self.position and current_signal != 0:
            if current_signal > 0:  # Buy signal
                self.buy(size=1)
                self.entry_bar = len(self)
                self.log_trade("BUY", 1, self.data.close[0])
            else:  # Sell signal
                self.sell(size=1)
                self.entry_bar = len(self)
                self.log_trade("SELL", 1, self.data.close[0])
    
    def close_position(self, reason):
        """Close current position and log the reason."""
        if self.position:
            self.close()
            self.log_trade(f"CLOSE ({reason})", 1, self.data.close[0])
    
    def log_trade(self, action, size, price):
        """Log trade information and calculate returns."""
        dt = self.data.datetime.datetime()
        
        trade_info = {
            'datetime': dt,
            'action': action,
            'size': size,
            'price': price,
            'rsi': self.rsi[0],
            'stoch_k': self.stoch.percK[0],
            'stoch_d': self.stoch.percD[0]
        }
        
        # Calculate trade P&L if it's a closing trade
        if "CLOSE" in action and len(self.trade_history) > 0:
            last_trade = self.trade_history[-1]
            if "BUY" in last_trade['action']:
                pnl = (price - last_trade['price']) * size
            else:
                pnl = (last_trade['price'] - price) * size
            trade_info['pnl'] = pnl
            
            # Store trade result by market regime
            self.store_regime_trade(trade_info)
        
        self.trade_history.append(trade_info)
        
        # Print trade information
        if "EXECUTED" in action or "CLOSE" in action:
            direction = "LONG" if "BUY" in action else "SHORT"
            print(f'TRADE: {dt.date()} {dt.time()} | {direction:5} | ${price:7.2f} | '
                  f'RSI: {self.rsi[0]:4.1f} | Stoch %K: {self.stoch.percK[0]:4.1f}')
    
    def store_regime_trade(self, trade_info):
        """Store trade results by market regime using only past data."""
        if len(self) < 20:  # Need enough bars for regime calculation
            return
            
        # Calculate historical volatility using past data only
        past_returns = []
        for i in range(1, 21):  # Last 20 bars
            if i < len(self):
                try:
                    ret = (self.data.close[-i] - self.data.close[-(i+1)]) / self.data.close[-(i+1)]
                    past_returns.append(ret)
                except:
                    continue
        
        vol = np.std(past_returns) if past_returns else 0
        median_vol = np.median([abs(r) for r in past_returns]) if past_returns else 0
        
        # Trend classification using SMA indicator with wider threshold
        try:
            trending = abs(self.data.close[0] - self.sma[0]) / self.sma[0] > 0.01
        except:
            trending = False
        
        # Store trade in appropriate regime buckets
        if vol > median_vol:
            self.regime_stats['high_vol']['trades'].append(trade_info)
        else:
            self.regime_stats['low_vol']['trades'].append(trade_info)
            
        if trending:
            self.regime_stats['trending']['trades'].append(trade_info)
        else:
            self.regime_stats['ranging']['trades'].append(trade_info)
    
    def stop(self):
        """Calculate and print final strategy statistics."""
        print('\n=== Data Analysis ===')
        print(f'Total Bars Processed: {self.total_bars}')
        print(f'Bars in Trading Sessions: {self.session_bars} ({self.session_bars/self.total_bars*100:.1f}%)')
        print(f'Bars with Signals: {self.signal_bars} ({self.signal_bars/self.session_bars*100:.1f}% of session bars)')
        
        if not self.trade_history:
            return
            
        print('\n=== Signal Quality Metrics ===')
        
        # Calculate Information Coefficient in a separate pass
        if len(self.signal_history) > 1:
            signals = pd.DataFrame(self.signal_history)
            
            # Calculate forward returns
            signals['next_return'] = signals['close'].shift(-1) / signals['close'] - 1
            signals = signals.dropna()  # Remove last row (no forward return)
            
            # Calculate IC only for periods with actual signals
            valid_signals = signals[signals['signal'] != 0]
            if len(valid_signals) > 0:
                ic = valid_signals['signal'].corr(valid_signals['next_return'])
                print(f'Information Coefficient: {ic:.3f}')
            else:
                print('No valid signals for IC calculation')
        
        # Calculate overall strategy metrics
        trades = pd.DataFrame([t for t in self.trade_history if 'pnl' in t])
        if len(trades) > 0:
            win_rate = len(trades[trades['pnl'] > 0]) / len(trades) * 100
            avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if len(trades[trades['pnl'] > 0]) > 0 else 0
            avg_loss = trades[trades['pnl'] < 0]['pnl'].mean() if len(trades[trades['pnl'] < 0]) > 0 else 0
            
            print(f'\nOverall Performance:')
            print(f'Total Trades: {len(trades)}')
            print(f'Win Rate: {win_rate:.1f}%')
            print(f'Avg Win: ${avg_win:.2f}')
            print(f'Avg Loss: ${avg_loss:.2f}')
            print(f'Profit Factor: {abs(avg_win/avg_loss):.2f}' if avg_loss != 0 else 'N/A')
            
            # Print regime analysis
            print('\nPerformance by Regime:')
            for regime, data in self.regime_stats.items():
                if len(data['trades']) > 0:
                    regime_pnl = sum([t['pnl'] for t in data['trades']])
                    regime_wins = len([t for t in data['trades'] if t['pnl'] > 0])
                    print(f'\n{regime.replace("_", " ").title()}:')
                    print(f'Trades: {len(data["trades"])}')
                    print(f'Win Rate: {(regime_wins/len(data["trades"])*100):.1f}%')
                    print(f'Total P&L: ${regime_pnl:.2f}')
