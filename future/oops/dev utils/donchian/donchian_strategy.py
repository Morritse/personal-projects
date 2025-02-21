import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class DonchianStrategy:
    def __init__(self, channel_period: int = 20, 
                 vol_lookback: int = 60, vol_target: float = 0.15,
                 stop_atr_multiple: float = 2.5):
        """
        Initialize strategy parameters
        
        Args:
            channel_period: Days for Donchian channel calculation (default: 20)
            vol_lookback: Days for volatility calculation (default: 60)
            vol_target: Target annualized volatility (default: 20%)
            stop_atr_multiple: Multiple of ATR for trailing stops (default: 2.5)
        """
        self.channel_period = channel_period
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target
        self.stop_atr_multiple = stop_atr_multiple

    def calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on Donchian channel breakouts.
        """
        # Calculate daily returns
        df['returns'] = df['Close'].pct_change()
        
        # Calculate Donchian channels
        df['upper_channel'] = df['High'].rolling(window=self.channel_period).max()
        df['lower_channel'] = df['Low'].rolling(window=self.channel_period).min()
        df['channel_mid'] = (df['upper_channel'] + df['lower_channel']) / 2
        
        # Calculate channel breakout signals
        df['long_breakout'] = df['Close'] > df['upper_channel'].shift(1)
        df['short_breakout'] = df['Close'] < df['lower_channel'].shift(1)
        
        # Generate position signals
        df['signal'] = 0  # Default to flat
        
        # Entry signals
        df.loc[df['long_breakout'], 'signal'] = 1
        df.loc[df['short_breakout'], 'signal'] = -1
        
        # Exit signals (crossing mid-channel in opposite direction)
        df.loc[(df['signal'].shift(1) > 0) & (df['Close'] < df['channel_mid']), 'signal'] = 0
        df.loc[(df['signal'].shift(1) < 0) & (df['Close'] > df['channel_mid']), 'signal'] = 0
        
        # Forward fill signals (maintain position until exit)
        df['signal'] = df['signal'].ffill()
        
        # Mark where signals are valid (after lookback period)
        df['signal_valid'] = False
        valid_start = self.channel_period + 1  # Need one extra day for signal calculation
        if valid_start < len(df):
            df.loc[df.index[valid_start:], 'signal_valid'] = True
        
        return df

    def calculate_position_sizes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position sizes based on volatility targeting"""
        
        # Calculate volatility (annualized)
        df['volatility'] = df['returns'].rolling(window=self.vol_lookback).std() * np.sqrt(252)
        
        # Calculate position sizes with volatility targeting
        df['position_size'] = self.vol_target / (df['volatility'] * np.sqrt(252))
        
        # Scale positions by signal
        df['position'] = df['position_size'] * df['signal']
        
        # Cap maximum position size at 1 (100% of capital)
        df['position'] = df['position'].clip(-1, 1)
        
        return df

    def apply_risk_management(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply trailing stops and other risk management rules.
        """
        # Calculate Average True Range (ATR)
        df['tr'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.vol_lookback).mean()
        
        # Calculate stop levels
        df['position_adj'] = df['position'].copy()
        df['prev_pos'] = df['position'].shift(1)
        
        # Long position stops
        long_stop = df['Close'].shift(1) - (self.stop_atr_multiple * df['atr'].shift(1))
        
        # Short position stops
        short_stop = df['Close'].shift(1) + (self.stop_atr_multiple * df['atr'].shift(1))
        
        # Apply stops using vectorized operations
        long_mask = df['prev_pos'] > 0
        short_mask = df['prev_pos'] < 0
        
        # Apply stops
        df.loc[long_mask & (df['Low'] < long_stop), 'position_adj'] = 0
        df.loc[short_mask & (df['High'] > short_stop), 'position_adj'] = 0
        
        # Update position to use stop-adjusted version
        df['position'] = df['position_adj']
        
        return df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate strategy returns and metrics"""
        
        # Calculate strategy returns (assuming next-day execution)
        df['strat_returns'] = df['position'].shift(1) * df['returns']
        
        # Calculate cumulative returns
        df['cum_returns'] = (1 + df['strat_returns']).cumprod()
        
        # Calculate rolling Sharpe ratio (annualized)
        df['rolling_sharpe'] = (
            df['strat_returns'].rolling(window=252).mean() * 252 /
            (df['strat_returns'].rolling(window=252).std() * np.sqrt(252))
        )
        
        return df

    def run_strategy(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete strategy and return results
        
        Returns:
            df: DataFrame with all signals and metrics
            metrics: Dict of strategy performance metrics
        """
        # Run all strategy components
        df = self.calculate_signals(df)
        df = self.calculate_position_sizes(df)
        df = self.apply_risk_management(df)
        df = self.calculate_returns(df)
        
        # Calculate strategy metrics
        metrics = {
            'total_return': df['cum_returns'].iloc[-1] - 1,
            'annual_return': df['strat_returns'].mean() * 252,
            'annual_vol': df['strat_returns'].std() * np.sqrt(252),
            'sharpe_ratio': df['strat_returns'].mean() * 252 / (df['strat_returns'].std() * np.sqrt(252)),
            'max_drawdown': (df['cum_returns'] / df['cum_returns'].cummax() - 1).min(),
            'win_rate': (df['strat_returns'] > 0).mean()
        }
        
        return df, metrics

def analyze_portfolio(dfs: Dict[str, pd.DataFrame], strategy: DonchianStrategy) -> pd.Series:
    """
    Analyze a portfolio of futures contracts
    
    Args:
        dfs: Dict of DataFrames containing price data for each instrument
        strategy: DonchianStrategy instance
    
    Returns:
        Portfolio level metrics
    """
    # Find common date range across all instruments
    common_index = None
    for df in dfs.values():
        if common_index is None:
            common_index = df.index
        else:
            common_index = common_index.intersection(df.index)
    
    # Initialize returns DataFrame
    all_returns = pd.DataFrame(index=common_index)
    
    # Run strategy on each instrument
    for symbol, df in dfs.items():
        # Align data to common index
        df_aligned = df.copy()
        df_aligned = df_aligned.loc[common_index]
        df_result, _ = strategy.run_strategy(df_aligned)
        # Only include returns where signals are valid
        valid_returns = np.where(df_result['signal_valid'], 
                               df_result['strat_returns'], 
                               np.nan).astype(float)
        all_returns[symbol] = pd.Series(valid_returns, index=all_returns.index, dtype=float)
    
    # Drop any rows where all returns are NaN
    all_returns = all_returns.dropna(how='all')
    
    if len(all_returns) == 0:
        raise ValueError("No valid returns data available for analysis")
    
    # Equal weight portfolio (1/N allocation)
    n_assets = len(all_returns.columns)
    portfolio_returns = all_returns.mean(axis=1)  # Simple average across assets
    
    # Scale portfolio volatility to match single-asset target
    portfolio_vol = portfolio_returns.std() * np.sqrt(252)
    scale_factor = strategy.vol_target / portfolio_vol
    portfolio_returns = portfolio_returns * scale_factor
    
    all_returns['portfolio'] = portfolio_returns
    
    # Calculate portfolio metrics
    portfolio_metrics = {
        'annual_return': all_returns['portfolio'].mean() * 252,
        'annual_vol': all_returns['portfolio'].std() * np.sqrt(252),
        'sharpe_ratio': (all_returns['portfolio'].mean() * 252) / 
                       (all_returns['portfolio'].std() * np.sqrt(252)),
        'max_drawdown': (
            (1 + all_returns['portfolio']).cumprod() / 
            (1 + all_returns['portfolio']).cumprod().cummax() - 1
        ).min()
    }
    
    return pd.Series(portfolio_metrics)
