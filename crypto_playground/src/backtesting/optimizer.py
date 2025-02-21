import itertools
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import os
from dotenv import load_dotenv

from data_handler import HistoricalDataHandler

@dataclass
class OptimizationResult:
    params: Dict
    profit_loss: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    avg_trade_duration: float

class StrategyOptimizer:
    def __init__(
        self,
        symbols: List[str] = ["BTC/USD", "ETH/USD"],
        lookback_days: int = 30,
        initial_capital: float = 100000.0
    ):
        # Initialize parameters
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.initial_capital = initial_capital
        
        # Initialize data handler
        self.data_handler = HistoricalDataHandler(
            symbols=symbols,
            lookback_days=lookback_days
        )
        
        # Parameter ranges for grid search
        self.param_grid = {
            'signal_thresholds': {
                'MIN_SIGNAL_STRENGTH': [0.01, 0.02, 0.05, 0.1],  # More lenient signal thresholds
                'MIN_CONFIDENCE': [0.1, 0.2, 0.3, 0.4],  # Lower confidence requirements
            },
            'risk_params': {
                'STOP_LOSS_ATR': [0.5, 1.0, 1.5, 2.0],  # Tighter stops
                'TAKE_PROFIT_ATR': [1.0, 1.5, 2.0, 2.5],  # Closer targets
            }
        }
        
    def _generate_param_combinations(self) -> List[Dict]:
        """Generate all parameter combinations for grid search."""
        param_combinations = []
        
        # Get all possible values for each parameter
        param_values = []
        param_names = []
        
        for category in self.param_grid.values():
            for param_name, values in category.items():
                param_names.append(param_name)
                param_values.append(values)
        
        # Generate all combinations
        for combination in itertools.product(*param_values):
            params = dict(zip(param_names, combination))
            param_combinations.append(params)
            
        return param_combinations
        
    def _calculate_metrics(
        self,
        trades: List[Dict],
        equity_curve: List[float]
    ) -> Tuple[float, float, float, float, int, float]:
        """Calculate performance metrics for a backtest run."""
        if not trades:
            return 0.0, 0.0, 0.0, 0.0, 0, 0.0
            
        # Calculate basic metrics
        profit_loss = (equity_curve[-1] / self.initial_capital - 1) * 100  # Convert to percentage
        winning_trades = len([t for t in trades if t['profit_loss'] > 0])
        win_rate = winning_trades / len(trades) if trades else 0
        
        # Calculate max drawdown
        peak = self.initial_capital
        max_drawdown = 0
        
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
            
        # Calculate Sharpe ratio (assuming risk-free rate of 0)
        returns = pd.Series([e2/e1 - 1 for e1, e2 in zip(equity_curve[:-1], equity_curve[1:])])
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if len(returns) > 1 else 0
        
        # Calculate average trade duration
        durations = [(t['exit_time'] - t['entry_time']).total_seconds() / 60 for t in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return profit_loss, win_rate, max_drawdown, sharpe_ratio, len(trades), avg_duration
        
    async def optimize(self, max_combinations: int = 100) -> List[OptimizationResult]:
        """Run grid search optimization."""
        # Get historical data with signals
        historical_data = await self.data_handler.fetch_data()
        
        # Generate parameter combinations
        all_combinations = self._generate_param_combinations()
        
        # Limit combinations if needed
        if len(all_combinations) > max_combinations:
            np.random.shuffle(all_combinations)
            combinations = all_combinations[:max_combinations]
        else:
            combinations = all_combinations
            
        results = []
        
        # Test each combination
        for params in combinations:
            try:
                # Run backtest with these parameters
                trades, equity_curve = await self._run_backtest(historical_data, params)
                
                # Calculate metrics
                profit_loss, win_rate, max_drawdown, sharpe_ratio, total_trades, avg_duration = \
                    self._calculate_metrics(trades, equity_curve)
                    
                # Only store results if we have valid metrics and trades
                if not any(np.isnan([profit_loss, win_rate, max_drawdown, sharpe_ratio])) and total_trades > 0:
                    # Create a copy of params to avoid reference issues
                    params_copy = params.copy()
                    
                    # Store result
                    result = OptimizationResult(
                        params=params_copy,
                        profit_loss=profit_loss,
                        win_rate=win_rate,
                        max_drawdown=max_drawdown,
                        sharpe_ratio=sharpe_ratio,
                        total_trades=total_trades,
                        avg_trade_duration=avg_duration
                    )
                    results.append(result)
                    
                    print(f"Valid result found - Sharpe: {sharpe_ratio:.2f}, "
                          f"PnL: {profit_loss:.2f}%, Trades: {total_trades}")
            except Exception as e:
                print(f"Error in parameter combination: {e}")
                continue
                
        # Sort by Sharpe ratio (only if we have results)
        if results:
            results.sort(key=lambda x: x.sharpe_ratio, reverse=True)
        else:
            print("Warning: No valid results found. Check data and parameters.")
        
        # Save results
        self._save_results(results)
        
        return results
        
    async def _run_backtest(
        self,
        historical_data: Dict[str, pd.DataFrame],
        params: Dict
    ) -> Tuple[List[Dict], List[float]]:
        """Run a single backtest with given parameters."""
        trades = []
        equity_curve = [self.initial_capital]
        current_equity = self.initial_capital
        open_positions = {}
        
        # Test each symbol
        for symbol, df in historical_data.items():
            # Skip first 20 bars to allow for indicator calculation
            for i in range(20, len(df)):
                current_bar = df.iloc[i]
                
                # Check for exit signals on open positions
                if symbol in open_positions:
                    position = open_positions[symbol]
                    
                    # Check stop loss and take profit
                    if position['side'] == 'long':
                        if current_bar['low'] <= position['stop_loss']:
                            # Calculate percentage P&L
                            pnl_pct = (position['stop_loss'] / position['entry_price'] - 1)
                            trade_pnl = current_equity * position['size'] * pnl_pct
                            current_equity += trade_pnl
                            
                            trades.append({
                                'symbol': symbol,
                                'entry_time': position['entry_time'],
                                'exit_time': current_bar.name,
                                'side': position['side'],
                                'entry_price': position['entry_price'],
                                'exit_price': position['stop_loss'],
                                'size': position['size'],
                                'profit_loss': pnl_pct,
                                'exit_reason': 'stop_loss'
                            })
                            del open_positions[symbol]
                            
                        elif current_bar['high'] >= position['take_profit']:
                            # Calculate percentage P&L
                            pnl_pct = (position['take_profit'] / position['entry_price'] - 1)
                            trade_pnl = current_equity * position['size'] * pnl_pct
                            current_equity += trade_pnl
                            
                            trades.append({
                                'symbol': symbol,
                                'entry_time': position['entry_time'],
                                'exit_time': current_bar.name,
                                'side': position['side'],
                                'entry_price': position['entry_price'],
                                'exit_price': position['take_profit'],
                                'size': position['size'],
                                'profit_loss': pnl_pct,
                                'exit_reason': 'take_profit'
                            })
                            del open_positions[symbol]
                    elif position['side'] == 'short':
                        if current_bar['high'] >= position['stop_loss']:
                            # Calculate percentage P&L
                            pnl_pct = -(position['stop_loss'] / position['entry_price'] - 1)
                            trade_pnl = current_equity * position['size'] * pnl_pct
                            current_equity += trade_pnl
                            
                            trades.append({
                                'symbol': symbol,
                                'entry_time': position['entry_time'],
                                'exit_time': current_bar.name,
                                'side': position['side'],
                                'entry_price': position['entry_price'],
                                'exit_price': position['stop_loss'],
                                'size': position['size'],
                                'profit_loss': pnl_pct,
                                'exit_reason': 'stop_loss'
                            })
                            del open_positions[symbol]
                            
                        elif current_bar['low'] <= position['take_profit']:
                            # Calculate percentage P&L
                            pnl_pct = -(position['take_profit'] / position['entry_price'] - 1)
                            trade_pnl = current_equity * position['size'] * pnl_pct
                            current_equity += trade_pnl
                            
                            trades.append({
                                'symbol': symbol,
                                'entry_time': position['entry_time'],
                                'exit_time': current_bar.name,
                                'side': position['side'],
                                'entry_price': position['entry_price'],
                                'exit_price': position['take_profit'],
                                'size': position['size'],
                                'profit_loss': pnl_pct,
                                'exit_reason': 'take_profit'
                            })
                            del open_positions[symbol]
                
                # Check for entry signals
                if symbol not in open_positions:
                    signal_strength = current_bar['combined_signal']
                    confidence = current_bar['confidence']
                    
                    # Long signal conditions
                    if signal_strength > params['MIN_SIGNAL_STRENGTH'] and confidence > params['MIN_CONFIDENCE']:
                        # Calculate position size as percentage of capital
                        pos_size = current_bar['position_size']
                        
                        # Open long position
                        open_positions[symbol] = {
                            'side': 'long',
                            'entry_price': current_bar['close'],
                            'stop_loss': current_bar['close'] - (current_bar['atr'] * params['STOP_LOSS_ATR']),
                            'take_profit': current_bar['close'] + (current_bar['atr'] * params['TAKE_PROFIT_ATR']),
                            'size': pos_size,
                            'entry_time': current_bar.name
                        }
                    
                    # Short signal conditions
                    elif signal_strength < -params['MIN_SIGNAL_STRENGTH'] and confidence > params['MIN_CONFIDENCE']:
                        # Calculate position size as percentage of capital
                        pos_size = current_bar['position_size']
                        
                        # Open short position
                        open_positions[symbol] = {
                            'side': 'short',
                            'entry_price': current_bar['close'],
                            'stop_loss': current_bar['close'] + (current_bar['atr'] * params['STOP_LOSS_ATR']),
                            'take_profit': current_bar['close'] - (current_bar['atr'] * params['TAKE_PROFIT_ATR']),
                            'size': pos_size,
                            'entry_time': current_bar.name
                        }
                
                # Calculate total equity including unrealized P&L
                total_equity = current_equity
                for pos in open_positions.values():
                    if pos['side'] == 'long':
                        pnl_pct = (current_bar['close'] / pos['entry_price'] - 1)
                    else:  # short
                        pnl_pct = -(current_bar['close'] / pos['entry_price'] - 1)
                    unrealized_pnl = total_equity * pos['size'] * pnl_pct
                    total_equity += unrealized_pnl
                
                equity_curve.append(total_equity)
                
        return trades, equity_curve
        
    def _save_results(self, results: List[OptimizationResult]):
        """Save optimization results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'optimization_results_{timestamp}.json'
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Convert results to JSON-serializable format
        results_data = []
        for result in results:
            result_dict = {
                'params': result.params,
                'metrics': {
                    'profit_loss': result.profit_loss,
                    'win_rate': result.win_rate,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_trades': result.total_trades,
                    'avg_trade_duration': result.avg_trade_duration
                }
            }
            results_data.append(result_dict)
            
        # Save to file
        with open(f'results/{filename}', 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"Results saved to results/{filename}")

if __name__ == '__main__':
    import asyncio
    
    async def main():
        optimizer = StrategyOptimizer()
        results = await optimizer.optimize()
        
        # Print top 5 results
        print("\nTop 5 Parameter Combinations:")
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"Profit/Loss: {result.profit_loss:.2f}%")
            print(f"Win Rate: {result.win_rate:.2f}%")
            print(f"Max Drawdown: {result.max_drawdown:.2f}%")
            print(f"Total Trades: {result.total_trades}")
            print(f"Avg Trade Duration: {result.avg_trade_duration:.1f} minutes")
            print("\nParameters:")
            for param, value in result.params.items():
                print(f"{param}: {value}")
                
    asyncio.run(main())
