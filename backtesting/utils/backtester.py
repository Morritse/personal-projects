import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from config import VERBOSE_DATA

@dataclass
class TradeResult:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    signal_strength: float
    trade_type: str  # 'LONG' or 'SHORT'

class Backtester:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        max_position_size: float = 0.2,  # 20% max per position
        commission: float = 0.001,  # 0.1% commission per trade
        verbose: bool = True  # Control trade output
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = max_position_size
        self.commission = commission
        self.verbose = verbose
        self.positions: Dict[str, Dict] = {}  # Current positions
        self.trades: List[TradeResult] = []  # Completed trades
        self.equity_curve: List[Tuple[datetime, float]] = []
        
    def calculate_position_size(self, symbol: str, signal_strength: float, price: float) -> float:
        """Calculate position size based on signal strength and available capital."""
        # Scale position size between 5-20% based on signal strength
        base_size = 0.05  # 5% minimum per position
        max_size = 0.20   # 20% maximum per position
        
        # Amplify signal strength for more aggressive sizing
        amplified_signal = min(1.0, abs(signal_strength) * 1.5)
        scaled_size = base_size + (amplified_signal * (max_size - base_size))
        position_size = min(scaled_size, max_size)
        
        # Calculate position size considering total positions
        max_positions = 5  # Maximum concurrent positions
        available_capital = self.current_capital * (1.0 / max(1, max_positions - len(self.positions)))
        
        # Scale by signal strength
        position_capital = available_capital * position_size
        shares = position_capital / price
        
        if VERBOSE_DATA and self.verbose:
            print(f"Position Sizing:")
            print(f"- Signal: {signal_strength:.3f} (amplified: {amplified_signal:.3f})")
            print(f"- Size %: {position_size:.1%}")
            print(f"- Capital: ${available_capital:,.2f}")
            print(f"- Shares: {shares:.2f}")
            
        return shares
        
    def enter_position(
        self,
        symbol: str,
        timestamp: datetime,
        price: float,
        signal_strength: float,
        trade_type: str
    ) -> bool:
        """Enter a new position."""
        if symbol in self.positions:
            if self.verbose:
                print(f"Already have position in {symbol}")
            return False
            
        # Calculate position size
        size = self.calculate_position_size(symbol, signal_strength, price)
        cost = size * price
        commission_cost = cost * self.commission
        
        # Check if we have enough capital
        if cost + commission_cost > self.current_capital:
            if self.verbose:
                print(f"Insufficient capital for {symbol} position")
            return False
            
        # Record position
        self.positions[symbol] = {
            'entry_time': timestamp,
            'entry_price': price,
            'size': size,
            'signal_strength': signal_strength,
            'trade_type': trade_type,
            'cost': cost + commission_cost
        }
        
        # Update capital
        self.current_capital -= (cost + commission_cost)
        if self.verbose:
            print(f"Position: {size:.2f} shares (${cost:,.2f})")
        return True
        
    def exit_position(
        self,
        symbol: str,
        timestamp: datetime,
        price: float
    ) -> Optional[TradeResult]:
        """Exit an existing position."""
        if symbol not in self.positions:
            if self.verbose:
                print(f"No position to exit in {symbol}")
            return None
            
        position = self.positions[symbol]
        proceeds = position['size'] * price
        commission_cost = proceeds * self.commission
        
        # Calculate P&L
        pnl = proceeds - position['cost'] - commission_cost
        if position['trade_type'] == 'SHORT':
            pnl = -pnl
            
        # Record trade result
        trade = TradeResult(
            symbol=symbol,
            entry_time=position['entry_time'],
            exit_time=timestamp,
            entry_price=position['entry_price'],
            exit_price=price,
            position_size=position['size'],
            pnl=pnl,
            signal_strength=position['signal_strength'],
            trade_type=position['trade_type']
        )
        
        # Update capital and records
        self.current_capital += proceeds - commission_cost
        self.trades.append(trade)
        del self.positions[symbol]
        
        if self.verbose:
            print(f"Result: {position['size']:.2f} shares, PnL: ${pnl:.2f} ({(pnl/position['cost'])*100:.1f}%)")
        return trade
        
    def update_equity(self, timestamp: datetime, prices: Dict[str, float]):
        """Update equity curve with current positions marked to market."""
        equity = self.current_capital
        
        # Add unrealized P&L from open positions
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_value = position['size'] * prices[symbol]
                if position['trade_type'] == 'LONG':
                    equity += current_value - position['cost']
                else:  # SHORT
                    equity += position['cost'] - current_value
                    
        self.equity_curve.append((timestamp, equity))
        
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_trade_pnl': 0.0,
                'avg_win_pnl': 0.0,
                'avg_loss_pnl': 0.0
            }
            
        # Calculate basic metrics
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t.pnl > 0])
        total_pnl = sum(t.pnl for t in self.trades)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average trade metrics
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        winning_trades = [t.pnl for t in self.trades if t.pnl > 0]
        losing_trades = [t.pnl for t in self.trades if t.pnl <= 0]
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Calculate returns
        equity_values = [e[1] for e in self.equity_curve]
        returns = pd.Series(equity_values).pct_change().dropna()
        
        # Calculate Sharpe Ratio (assuming daily data)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if len(returns) > 1 else 0
        
        # Maximum drawdown
        peak = equity_values[0]
        max_dd = 0
        for value in equity_values[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'return_pct': (self.current_capital - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'avg_trade_pnl': avg_trade_pnl,
            'avg_win_pnl': avg_win,
            'avg_loss_pnl': avg_loss
        }
