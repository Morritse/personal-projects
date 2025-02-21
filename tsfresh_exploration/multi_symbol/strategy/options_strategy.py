import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from vwap_obv_strategy_vec import VWAPOBVCrossoverVec

class OptionsStrategy:
    def __init__(self, api_key=None, api_secret=None, params=None, backtest_mode=False):
        if not backtest_mode:
            self.client = StockHistoricalDataClient(api_key, api_secret, raw_data=True)
        
        # Default strategy parameters (optimized)
        self.params = {
            'vwap_length': 50,         # Optimal from backtest
            'mfi_length': 9,           # Optimal from backtest
            'mfi_oversold': 30,        # Optimal from backtest
            'mfi_overbought': 70,      # Optimal from backtest
            'regime_window': 20,
            'vol_percentile': 67,
        } if params is None else params
        
        # Options-specific parameters
        self.options_params = {
            'min_dte': 5,              # Minimum days to expiry
            'max_dte': 21,             # Maximum days to expiry
            'delta_target': 0.30,      # Target delta for strike selection
            'max_position': 0.25,      # Maximum position size (% of account)
            'iv_rank_min': 0.20,       # Minimum IV rank to trade
            'rel_vol_lookback': 252,   # Lookback for relative volatility
            'max_hold_hours': 24,      # Optimal from backtest
            'profit_target': 0.02,     # 2% profit target
            'stop_loss': 0.02         # 2% stop loss
        }
        
        self.base_strategy = VWAPOBVCrossoverVec(self.params)
    
    def calculate_historical_volatility(self, df, window=20):
        """Calculate historical volatility."""
        returns = df['close'].pct_change()
        hist_vol = returns.rolling(window=window).std() * np.sqrt(252)
        return hist_vol
    
    def calculate_relative_volatility(self, df, window=20):
        """Calculate volatility relative to its history."""
        hist_vol = self.calculate_historical_volatility(df, window)
        vol_percentile = hist_vol.rolling(self.options_params['rel_vol_lookback']).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        return vol_percentile
    
    def determine_market_regime(self, df):
        """Determine market regime using multiple factors."""
        # Calculate returns and volatility
        returns = df['close'].pct_change()
        vol = returns.rolling(window=self.params['regime_window']).std() * np.sqrt(252)
        ret = returns.rolling(window=self.params['regime_window']).mean() * 252
        
        # Calculate relative metrics
        vol_rank = vol.rolling(self.options_params['rel_vol_lookback']).rank(pct=True)
        ret_rank = ret.rolling(self.options_params['rel_vol_lookback']).rank(pct=True)
        
        # Create regime series
        regime = pd.Series('none', index=df.index)
        
        # Define regimes based on relative metrics
        bull_mask = (ret_rank > 0.6) & (vol_rank > 0.4)  # Relaxed conditions
        bear_mask = (ret_rank < 0.4) & (vol_rank > 0.4)
        
        regime[bull_mask] = 'bull_high_vol'
        regime[bear_mask] = 'bear_high_vol'
        
        return regime
    
    def select_strike(self, price, hist_vol, rel_vol, direction='long'):
        """Select option strike based on volatility metrics."""
        daily_vol = hist_vol / np.sqrt(252)
        
        if direction == 'long':
            # Strike selection based on relative volatility
            if rel_vol > 0.7:  # High relative vol
                strike_distance = 0.5 * daily_vol
            elif rel_vol > 0.3:  # Medium relative vol
                strike_distance = 0.75 * daily_vol
            else:  # Low relative vol
                strike_distance = 1.0 * daily_vol
            
            strike = price * (1 + strike_distance)
        else:
            if rel_vol > 0.7:
                strike_distance = 0.5 * daily_vol
            elif rel_vol > 0.3:
                strike_distance = 0.75 * daily_vol
            else:
                strike_distance = 1.0 * daily_vol
            
            strike = price * (1 - strike_distance)
        
        return round(strike, 2)
    
    def get_optimal_expiry(self, hist_vol, rel_vol):
        """Select optimal expiry based on volatility metrics."""
        if rel_vol > 0.7:  # High relative vol
            dte = max(5, min(7, self.options_params['max_dte']))
        elif rel_vol > 0.3:  # Medium relative vol
            dte = max(10, min(14, self.options_params['max_dte']))
        else:  # Low relative vol
            dte = max(14, min(21, self.options_params['max_dte']))
        
        return dte
    
    def calculate_position_size(self, account_size, volatility, rel_vol):
        """Calculate position size based on volatility and IV rank."""
        # Base size from Kelly simulation
        base_size = account_size * 0.093  # Quarter Kelly from simulation
        
        # Adjust for volatility
        if volatility > 0.50:  # High vol
            vol_adjustment = 0.7  # Reduce size
        else:
            vol_adjustment = 1.0
        
        # Adjust for relative volatility
        if rel_vol > 0.7:  # High relative vol
            rel_vol_adjustment = 0.7  # Reduce size
        else:
            rel_vol_adjustment = 1.0
        
        # Calculate final size
        position_size = base_size * vol_adjustment * rel_vol_adjustment
        
        # Cap at max position size
        max_size = account_size * self.options_params['max_position']
        return min(position_size, max_size)
    
    def run(self, symbol, account_size=100000, lookback_days=252):
        """Run strategy on live data."""
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date,
            adjustment=Adjustment.ALL
        )
        
        print(f"\nAnalyzing {symbol}...")
        bars = self.client.get_stock_bars(request)
        df = pd.DataFrame(bars[symbol])
        
        # Prepare data
        column_map = {
            'c': 'close',
            'h': 'high',
            'l': 'low',
            'o': 'open',
            'v': 'volume',
            't': 'timestamp'
        }
        df = df.rename(columns=column_map)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Calculate indicators
        df['vwap'] = self.base_strategy.calculate_vwap(df, self.params['vwap_length'])
        df['mfi'] = self.base_strategy.calculate_mfi(df, self.params['mfi_length'])
        df['obv'] = self.base_strategy.calculate_obv(df)
        df['obv_change'] = df['obv'].diff()
        df['hist_vol'] = self.calculate_historical_volatility(df)
        df['rel_vol'] = self.calculate_relative_volatility(df)
        df['regime'] = self.determine_market_regime(df)
        
        # Get current metrics
        current_price = df['close'].iloc[-1]
        current_vwap = df['vwap'].iloc[-1]
        current_mfi = df['mfi'].iloc[-1]
        current_obv_change = df['obv_change'].iloc[-1]
        current_regime = df['regime'].iloc[-1]
        current_vol = df['hist_vol'].iloc[-1]
        current_rel_vol = df['rel_vol'].iloc[-1]
        
        # Check for entry conditions
        entry_signal = (
            (current_regime != 'none' or current_rel_vol > 0.4) and
            current_mfi < self.params['mfi_oversold'] and
            current_price < current_vwap and
            current_obv_change < 0
        )
        
        if entry_signal:
            # Calculate trade parameters
            strike = self.select_strike(current_price, current_vol, current_rel_vol)
            dte = self.get_optimal_expiry(current_vol, current_rel_vol)
            size = self.calculate_position_size(account_size, current_vol, current_rel_vol)
            
            return {
                'symbol': symbol,
                'price': current_price,
                'strike': strike,
                'dte': dte,
                'position_size': size,
                'hist_vol': current_vol,
                'rel_vol': current_rel_vol,
                'regime': current_regime
            }
        
        return None

if __name__ == "__main__":
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    strategy = OptionsStrategy(
        config['alpaca']['api_key'],
        config['alpaca']['api_secret']
    )
    
    # Test on a few symbols
    test_symbols = ['NVDA', 'META', 'AAPL']
    
    for symbol in test_symbols:
        try:
            trade = strategy.run(symbol)
            if trade:
                print(f"\nFound setup for {symbol}!")
                print(f"Strike: ${trade['strike']:.2f}")
                print(f"DTE: {trade['dte']}")
                print(f"Size: ${trade['position_size']:.2f}")
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
