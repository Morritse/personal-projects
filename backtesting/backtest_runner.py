import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from tqdm import tqdm

from backtest_strategy import BacktestStrategy
from utils.backtester import Backtester
from utils.optimizer import StrategyOptimizer
from utils.data_handler import BacktestDataHandler
from utils.indicators_cache import IndicatorCache
from config import VERBOSE_DATA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestRunner:
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframes: List[str] = ['short', 'medium', 'long'],
        initial_capital: float = 100000.0,
        commission: float = 0.001
    ):
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.timeframes = timeframes
        self.initial_capital = initial_capital
        self.commission = commission
        
        # Initialize components (cache data handler)
        self._data_fetcher = None
        self.indicator_cache = IndicatorCache()
        
        # Create results directory
        self.results_dir = Path('backtesting/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    @property
    def data_fetcher(self):
        """Lazy initialize data fetcher."""
        if self._data_fetcher is None:
            self._data_fetcher = BacktestDataHandler()
        return self._data_fetcher
        
    async def fetch_historical_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch historical data for all symbols and timeframes."""
        print("\nFetching historical data...")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}\n")
        
        # Fetch data for all symbols with progress bar
        self.data = {}
        for symbol in tqdm(self.symbols, desc="Fetching Data"):
            symbol_data = await self.data_fetcher.fetch_historical_data(
                symbols=[symbol],
                start_date=self.start_date,
                end_date=self.end_date
            )
            if symbol_data:
                self.data.update(symbol_data)
        
        if not self.data:
            logger.error("No data fetched for any symbols")
            return {}
            
        if VERBOSE_DATA:
            print(f"\nFetched data for {len(self.data)} symbols")
            for symbol, timeframe_data in self.data.items():
                print(f"\n{symbol}:")
                for timeframe, df in timeframe_data.items():
                    print(f"- {timeframe}: {len(df)} bars")
                    
        return self.data
        
    def run_backtest(
        self,
        strategy_params: Dict = None,
        save_results: bool = True
    ) -> Dict:
        """Run backtest with given parameters."""
        # Reset backtester for new run
        backtester = Backtester(
            initial_capital=self.initial_capital,
            commission=self.commission
        )
        
        # Initialize strategy with parameters
        strategy = BacktestStrategy(
            symbols=self.symbols,
            **strategy_params if strategy_params else {}
        )
        
        # Pre-calculate all indicators with progress tracking
        print("\nPre-calculating indicators...")
        total_calculations = len(self.data) * len(self.timeframes)
        with tqdm(total=total_calculations, desc="Calculating Indicators") as pbar:
            self.indicator_cache.precalculate_indicators(self.data, progress_callback=lambda: pbar.update(1))
        
        # Run backtest chronologically
        logger.info("Running backtest...")
        
        # Get all timestamps from short timeframe data
        all_timestamps = set()
        for symbol_data in self.data.values():
            if 'short' in symbol_data and not symbol_data['short'].empty:
                all_timestamps.update(symbol_data['short'].index)
        
        if not all_timestamps:
            logger.error("No valid timestamps found")
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
            
        # Sort timestamps and sample at 5-minute intervals
        timestamps = sorted(all_timestamps)
        sampled_timestamps = []
        last_processed = None
        
        for timestamp in timestamps:
            # Only process every 5 minutes
            if last_processed is None or (timestamp - last_processed).total_seconds() >= 300:  # 5 minutes = 300 seconds
                sampled_timestamps.append(timestamp)
                last_processed = timestamp
        
        # Process sampled timestamps with detailed progress bar
        total_minutes = len(sampled_timestamps) * 5  # 5 minutes per sample
        total_hours = total_minutes / 60
        start_time = sampled_timestamps[0]
        end_time = sampled_timestamps[-1]
        
        print(f"\nBacktesting {total_hours:.1f} hours of market data")
        print(f"Period: {start_time} to {end_time}")
        print(f"Samples: {len(sampled_timestamps)} ({5} minute intervals)\n")
        
        with tqdm(total=len(sampled_timestamps), desc="Processing Market Data") as pbar:
            for timestamp in sampled_timestamps:
                try:
                    # Get current prices
                    current_prices = {}
                    for symbol, timeframes in self.data.items():
                        if 'short' in timeframes:
                            mask = timeframes['short'].index <= timestamp
                            if mask.any():
                                current_prices[symbol] = timeframes['short'].loc[mask, 'close'].iloc[-1]
                    
                    # Get signals for this timestamp
                    signals = self.indicator_cache.get_signals_at_timestamp(timestamp)
                    
                    # Run strategy
                    if signals and current_prices:
                        strategy.backtest(
                            backtester=backtester,
                            signals=signals,
                            timestamp=timestamp,
                            prices=current_prices
                        )
                except Exception as e:
                    logger.error(f"Error processing timestamp {timestamp}: {str(e)}")
                finally:
                    pbar.update(1)
                    # Update progress description with current time
                    pbar.set_description(f"Testing {timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        # Close any remaining positions
        if current_prices:
            strategy.close_all_positions(
                backtester=backtester,
                timestamp=sampled_timestamps[-1],
                prices=current_prices
            )
        
        # Get results
        metrics = backtester.get_performance_metrics()
        logger.info(f"Backtest completed. Metrics: {metrics}")
        
        if save_results:
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.results_dir / f'backtest_results_{timestamp}.json'
            
            # Combine metrics with parameters for easier comparison
            metrics.update({
                'parameters': strategy_params if strategy_params else {}
            })
            
            # Save full results
            results = {
                'parameters': strategy_params if strategy_params else {},
                'metrics': metrics,
                'trades': [{
                    'symbol': trade.symbol,
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat(),
                    'entry_price': float(trade.entry_price),
                    'exit_price': float(trade.exit_price),
                    'position_size': float(trade.position_size),
                    'pnl': float(trade.pnl),
                    'signal_strength': float(trade.signal_strength),
                    'trade_type': trade.trade_type
                } for trade in backtester.trades],
                'equity_curve': [
                    {'timestamp': ts.isoformat(), 'equity': float(eq)}
                    for ts, eq in backtester.equity_curve
                ]
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            logger.info(f"Results saved to {results_file}")
            
        return metrics
        
    async def optimize_parameters(
        self,
        param_ranges: Dict[str, tuple],
        n_iterations: int = 100,
        n_splits: int = 3
    ) -> Dict:
        """Optimize strategy parameters using walk-forward optimization."""
        print("\n" + "="*50)
        print("OPTIMIZATION CONFIGURATION")
        print("="*50)
        print(f"Parameters to optimize: {', '.join(param_ranges.keys())}")
        print(f"Points per parameter: {n_iterations}")
        print(f"Total combinations: {n_iterations**len(param_ranges)}")
        print(f"Time periods: {n_splits}")
        print("="*50 + "\n")
        
        # Initialize optimizer
        # Initialize data handler with verbose=False for optimization
        self._data_fetcher = BacktestDataHandler(verbose=False)
        
        optimizer = StrategyOptimizer(
            strategy_class=BacktestStrategy,
            param_space=param_ranges,
            data=self.data,
            n_iterations=n_iterations,
            n_splits=n_splits
        )
        
        # Run optimization
        print("Starting parameter optimization...")
        results = await optimizer.optimize()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'optimization_results_{timestamp}.json'
        optimizer.save_results(results_file)
        
        # Generate plots
        plot_file = self.results_dir / f'optimization_plots_{timestamp}.png'
        optimizer.plot_optimization_results(save_path=str(plot_file))
        
        logger.info(f"Optimization completed. Best parameters: {results['best_params']}")
        logger.info(f"Results saved to {results_file}")
        
        return results
