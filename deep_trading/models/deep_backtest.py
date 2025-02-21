"""
Deep Learning Model Backtester
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict
import os
import pickle
from datetime import datetime
import tensorflow as tf
from .deep_model import DeepTradingModel
from ..utils.deep_data_utils import prepare_features_for_prediction, clean_features, validate_features

class DeepModelBacktester:
    def __init__(self, model, symbol: str, risk_free_rate: float = 0.02):
        """
        Initialize backtester.
        
        Args:
            model: Trained TensorFlow model
            symbol: Trading symbol
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.model = model
        self.symbol = symbol
        self.risk_free_rate = risk_free_rate
        
    def calculate_market_regimes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime features."""
        # Market trend regime
        data['market_trend'] = data['close'].pct_change(20)
        data['market_regime'] = pd.qcut(
            data['market_trend'], 
            q=5, 
            labels=['strong_down', 'down', 'neutral', 'up', 'strong_up']
        )
        
        # Technical regime based on moving averages
        data['tech_trend'] = (data['close'] - data['sma_50']) / data['sma_50']
        data['tech_regime'] = pd.qcut(
            data['tech_trend'], 
            q=5, 
            labels=['tech_strong_down', 'tech_down', 'tech_neutral', 'tech_up', 'tech_strong_up']
        )
        
        # Volatility regime
        data['vol_trend'] = data['atr'].rolling(10).mean() / data['close']
        data['volatility_regime'] = pd.qcut(
            data['vol_trend'], 
            q=5, 
            labels=['very_low', 'low', 'normal', 'high', 'very_high']
        )
        
        # Bear market regime based on drawdown from recent high
        data['rolling_max'] = data['close'].rolling(60).max()
        data['drawdown'] = (data['close'] - data['rolling_max']) / data['rolling_max']
        data['bear_regime'] = pd.qcut(
            data['drawdown'], 
            q=5, 
            labels=['strong_bull', 'bull', 'neutral', 'bear', 'strong_bear']
        )
        
        # Rates regime based on moving average trends
        data['rates_trend'] = data['sma_20'].pct_change(20)
        data['rates_regime'] = pd.qcut(
            data['rates_trend'], 
            q=5, 
            labels=['rates_up_strong', 'rates_up', 'rates_neutral', 'rates_down', 'rates_down_strong']
        )
        
        return data
        
    def backtest(self, data: pd.DataFrame, initial_capital: float = 100000.0,
                position_size: float = 0.2) -> Tuple[pd.DataFrame, Dict]:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            initial_capital: Starting capital
            position_size: Size of each position as fraction of capital
            
        Returns:
            Tuple of (trades DataFrame, performance metrics)
        """
        trades = []
        capital = initial_capital
        position = None
        max_capital = capital
        min_capital = capital
        
        # Trading thresholds
        ENTRY_THRESHOLD = 0.55  # Increased confidence requirement
        EXIT_THRESHOLD = 0.45   # More conservative exit
        STOP_LOSS = 0.03       # Tighter stop loss
        TAKE_PROFIT = 0.06     # More realistic take profit
        MIN_HOLDING_DAYS = 3   # Shorter minimum hold
        MAX_HOLDING_DAYS = 20  # Shorter maximum hold
        
        # Clean and validate data
        data = clean_features(data)
        if not validate_features(data, self.symbol):
            raise ValueError(f"Data validation failed for {self.symbol}")
        
        # Add market regime features
        data = self.calculate_market_regimes(data)
        
        # Prepare sequences for prediction
        seq_length = 60
        for i in range(seq_length, len(data)):
            sequence = data.iloc[i-seq_length:i]
            current_price = data.iloc[i]['close']
            
            # Get technical signals
            sma_signal = data.iloc[i]['sma_20'] > data.iloc[i]['sma_50']
            momentum_signal = data.iloc[i]['returns'] > 0
            volatility_signal = data.iloc[i]['atr'] < data.iloc[i]['atr'].rolling(20).mean()
            trend_signal = current_price > data.iloc[i]['sma_50']
            
            # Dynamic stop loss based on ATR
            dynamic_stop = data.iloc[i]['atr'] / current_price * 2  # 2x ATR for stop loss
            
            try:
                # Prepare features and make prediction
                features = sequence.copy()
                features = features.select_dtypes(include=[np.number])  # Keep only numeric columns
                features = features.values.reshape(1, seq_length, -1)  # Reshape for model
                features = tf.cast(features, tf.float32)  # Convert to float32
                
                predictions = self.model.predict(features, verbose=0)
                return_pred = predictions[0][0][0]  # First output (returns)
                direction_pred = predictions[1][0][0]  # Second output (direction)
                
            except Exception as e:
                print(f"Error making prediction: {str(e)}")
                continue
            
            # Trading logic
            if position is None:  # No position
                # Enter if model and technical signals align
                signal_strength = (sma_signal + momentum_signal + volatility_signal + trend_signal) / 4
                if direction_pred > ENTRY_THRESHOLD and signal_strength >= 0.5:
                    position_size_usd = capital * position_size
                    shares = position_size_usd / current_price
                    position = {
                        'type': 'long',
                        'entry_price': current_price,
                        'shares': shares,
                        'entry_date': data.index[i],
                        'stop_loss_price': current_price * (1 - max(STOP_LOSS, dynamic_stop)),
                        'take_profit_price': current_price * (1 + TAKE_PROFIT)
                    }
                    trades.append({
                        'date': data.index[i],
                        'type': 'buy',
                        'price': current_price,
                        'shares': shares,
                        'value': position_size_usd,
                        'capital': capital,
                        'signal': direction_pred,
                        'signal_strength': signal_strength,
                        'market_regime': data.iloc[i]['market_regime'],
                        'tech_regime': data.iloc[i]['tech_regime'],
                        'volatility_regime': data.iloc[i]['volatility_regime'],
                        'bear_regime': data.iloc[i]['bear_regime'],
                        'rates_regime': data.iloc[i]['rates_regime']
                    })
                    
            else:  # In position
                # Calculate current position value
                current_value = position['shares'] * current_price
                capital_with_position = capital + (current_value - (position['shares'] * position['entry_price']))
                max_capital = max(max_capital, capital_with_position)
                min_capital = min(min_capital, capital_with_position)
                
                # Calculate holding period
                holding_days = (data.index[i] - position['entry_date']).days
                
                # Exit conditions
                exit_signal = direction_pred < EXIT_THRESHOLD
                stop_loss = current_price <= position['stop_loss_price']
                take_profit = current_price >= position['take_profit_price']
                momentum_exit = not momentum_signal and not trend_signal
                time_exit = holding_days >= MAX_HOLDING_DAYS
                
                if (exit_signal or stop_loss or take_profit or momentum_exit or time_exit) and holding_days >= MIN_HOLDING_DAYS:
                    exit_value = position['shares'] * current_price
                    profit = exit_value - (position['shares'] * position['entry_price'])
                    capital = capital_with_position
                    
                    trades.append({
                        'date': data.index[i],
                        'type': 'sell',
                        'price': current_price,
                        'shares': position['shares'],
                        'value': exit_value,
                        'profit': profit,
                        'capital': capital,
                        'signal': direction_pred,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'momentum_exit': momentum_exit,
                        'time_exit': time_exit,
                        'return': profit / (position['shares'] * position['entry_price']),
                        'holding_period': holding_days,
                        'market_regime': data.iloc[i]['market_regime'],
                        'tech_regime': data.iloc[i]['tech_regime'],
                        'volatility_regime': data.iloc[i]['volatility_regime'],
                        'bear_regime': data.iloc[i]['bear_regime'],
                        'rates_regime': data.iloc[i]['rates_regime']
                    })
                    position = None
        
        # Close any open position at the end
        if position is not None:
            exit_value = position['shares'] * data.iloc[-1]['close']
            profit = exit_value - (position['shares'] * position['entry_price'])
            capital += profit
            holding_days = (data.index[-1] - position['entry_date']).days
            
            trades.append({
                'date': data.index[-1],
                'type': 'sell',
                'price': data.iloc[-1]['close'],
                'shares': position['shares'],
                'value': exit_value,
                'profit': profit,
                'capital': capital,
                'signal': 0,  # End of period exit
                'stop_loss': False,
                'take_profit': False,
                'momentum_exit': False,
                'time_exit': True,
                'return': profit / (position['shares'] * position['entry_price']),
                'holding_period': holding_days,
                'market_regime': data.iloc[-1]['market_regime'],
                'tech_regime': data.iloc[-1]['tech_regime'],
                'volatility_regime': data.iloc[-1]['volatility_regime'],
                'bear_regime': data.iloc[-1]['bear_regime'],
                'rates_regime': data.iloc[-1]['rates_regime']
            })
        
        # Calculate metrics
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            sell_trades = trades_df[trades_df['type'] == 'sell']
            winning_trades = sell_trades[sell_trades['profit'] > 0]
            win_rate = len(winning_trades) / len(sell_trades) if len(sell_trades) > 0 else 0
            
            # Calculate max drawdown
            max_drawdown = (max_capital - min_capital) / max_capital if max_capital > initial_capital else 0
            
            # Calculate Sharpe ratio
            if len(sell_trades) > 0:
                returns = sell_trades['return'].values
                excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
                sharpe_ratio = np.sqrt(252) * (np.mean(excess_returns) / np.std(returns)) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
            
            # Calculate average holding period
            avg_holding_period = sell_trades['holding_period'].mean() if len(sell_trades) > 0 else 0
            
            metrics = {
                'Initial Capital': initial_capital,
                'Final Value': capital,
                'Total Return': (capital - initial_capital) / initial_capital,
                'Max Drawdown': max_drawdown,
                'Total Trades': len(sell_trades),
                'Win Rate': win_rate,
                'Sharpe Ratio': sharpe_ratio,
                'Stop Loss Hits': len(sell_trades[sell_trades['stop_loss']]),
                'Take Profit Hits': len(sell_trades[sell_trades['take_profit']]),
                'Momentum Exits': len(sell_trades[sell_trades['momentum_exit']]),
                'Time Exits': len(sell_trades[sell_trades['time_exit']]),
                'Avg Holding Period': avg_holding_period,
                'Avg Return Per Trade': np.mean(returns) if len(sell_trades) > 0 else 0
            }
        else:
            metrics = {
                'Initial Capital': initial_capital,
                'Final Value': capital,
                'Total Return': 0,
                'Max Drawdown': 0,
                'Total Trades': 0,
                'Win Rate': 0,
                'Sharpe Ratio': 0,
                'Stop Loss Hits': 0,
                'Take Profit Hits': 0,
                'Momentum Exits': 0,
                'Time Exits': 0,
                'Avg Holding Period': 0,
                'Avg Return Per Trade': 0
            }
        
        return trades_df, metrics
