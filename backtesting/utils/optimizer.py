import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import asyncio
from typing import Dict, List, Tuple, Coroutine
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import itertools
import multiprocessing

from utils.backtester import Backtester
from utils.indicators_cache import IndicatorCache
from utils.data_handler import BacktestDataHandler

class StrategyOptimizer:
    def __init__(
        self,
        strategy_class,
        param_space: Dict[str, Tuple[float, float]],
        data: Dict[str, Dict[str, pd.DataFrame]],
        n_splits: int = 5,
        n_iterations: int = 100,
        random_state: int = 42
    ):
        """Initialize the optimizer."""
        self.strategy_class = strategy_class
        self.data = data
        self.n_splits = n_splits
        self.n_iterations = n_iterations
        self.param_space = param_space
        self.random_state = random_state
        
        # Initialize components with verbose=False to suppress output during optimization
        self.indicator_cache = IndicatorCache()
        self._data_fetcher = BacktestDataHandler(verbose=False)
        
        # Store results
        self.best_params = None
        self.best_score = float('-inf')
        self.cv_results = []
        
        # Pre-calculate data that won't change between parameter sets
        print("\nInitializing optimization...")
        print("Pre-calculating data for optimization...")
        self.indicator_cache.precalculate_indicators(self.data)
        
        # Cache timestamps and prices
        print("Caching timestamps and prices...")
        self.all_timestamps = set()
        self.price_cache = {}
        for symbol, timeframe_data in self.data.items():
            if 'short' in timeframe_data and not timeframe_data['short'].empty:
                self.all_timestamps.update(timeframe_data['short'].index)
                # Cache prices for each timestamp
                for timestamp in timeframe_data['short'].index:
                    if timestamp not in self.price_cache:
                        self.price_cache[timestamp] = {}
                    self.price_cache[timestamp][symbol] = timeframe_data['short'].loc[timestamp, 'close']
        
        self.timestamps = sorted(self.all_timestamps)
        if not self.timestamps:
            raise ValueError("No valid timestamps found in data")
            
        # Pre-calculate time splits
        print("Pre-calculating time splits...")
        cv = TimeSeriesSplit(n_splits=self.n_splits)
        self.time_splits = list(cv.split(self.timestamps))
        
        # Set batch size based on CPU cores
        self.batch_size = max(10, multiprocessing.cpu_count() * 2)
        print(f"Using batch size of {self.batch_size} for parallel processing")
        
    async def objective_function(self, params: Dict, time_pbar: tqdm = None) -> float:
        """Objective function to minimize/maximize."""
        cv_scores = []
        
        # Use pre-calculated time splits
        for train_idx, test_idx in self.time_splits:
            try:
                # Initialize strategy and backtester (with verbose=False to suppress trade output)
                strategy = self.strategy_class(
                    symbols=list(self.data.keys()),
                    verbose=False,  # Suppress trade output during optimization
                    **params
                )
                backtester = Backtester(
                    initial_capital=100000.0,
                    max_position_size=0.1,
                    commission=0.001,
                    verbose=False  # Suppress trade output during optimization
                )
                
                # Process each timestamp in test period
                test_timestamps = [self.timestamps[i] for i in test_idx]
                for timestamp in test_timestamps:
                    # Use cached prices
                    current_prices = self.price_cache[timestamp]
                    
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
                
                # Calculate metrics
                metrics = backtester.get_performance_metrics()
                
                # Calculate composite score
                sharpe = metrics.get('sharpe_ratio', 0.0)
                win_rate = metrics.get('win_rate', 0.0)
                max_dd = metrics.get('max_drawdown', 1.0)
                total_trades = metrics.get('total_trades', 0)
                
                if total_trades < 5:  # Require minimum trades
                    score = 0.0
                else:
                    score = sharpe * (1 + win_rate) * (1 - max_dd)
                    
                cv_scores.append(score)
                if time_pbar:
                    time_pbar.update(1)
                
            except Exception as e:
                print(f"Error in optimization fold: {str(e)}")
                cv_scores.append(0.0)
            
        return np.mean(cv_scores)
        
    async def optimize(self) -> Dict:
        """Run grid search optimization with parallel processing."""
        # Generate grid points for each parameter
        param_grids = {}
        for param, (low, high) in self.param_space.items():
            # Use n_iterations points per parameter
            param_grids[param] = np.linspace(low, high, self.n_iterations)
            
        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())
        combinations = list(itertools.product(*param_values))
        total_sets = len(combinations)
        
        print("\n" + "="*80)
        print("OPTIMIZATION PROGRESS")
        print("="*80)
        print(f"Testing {total_sets} parameter combinations across {self.n_splits} time periods")
        print(f"Parameters being optimized:")
        for param, (low, high) in self.param_space.items():
            print(f"- {param}: {low:.3f} to {high:.3f}")
        print(f"Using {self.batch_size} parallel processes")
        print("="*80 + "\n")
        
        # Create batches of combinations for parallel processing
        batches = [combinations[i:i + self.batch_size] for i in range(0, len(combinations), self.batch_size)]
        
        # Process batches asynchronously with nested progress bars
        with tqdm(total=total_sets, desc="Parameter Sets", position=0) as param_pbar:
            with tqdm(total=self.n_splits, desc="Time Periods", position=1, leave=True) as time_pbar:
                for batch_num, batch in enumerate(batches):
                    tasks = []
                    for values in batch:
                        params = dict(zip(param_names, values))
                        current_set = len(self.cv_results) + 1
                        param_str = ", ".join(f"{k}={v:.3f}" for k,v in params.items())
                        print(f"\nBatch {batch_num + 1}/{len(batches)}: Testing Set {current_set}/{total_sets}")
                        print(f"Parameters: {param_str}")
                        tasks.append(self.objective_function(params, time_pbar))
                    
                    # Wait for batch to complete
                    batch_scores = await asyncio.gather(*tasks)
                    
                    # Process batch results
                    for values, score in zip(batch, batch_scores):
                        params = dict(zip(param_names, values))
                        self.cv_results.append({
                            'params': params,
                            'score': score
                        })
                        
                        # Update best if improved
                        if score > self.best_score:
                            self.best_score = score
                            self.best_params = params
                            param_str = ", ".join(f"{k}={v:.3f}" for k,v in params.items())
                            print(f"\n[{len(self.cv_results)}/{total_sets}] New best score: {score:.4f}")
                            print(f"Parameters: {param_str}")
                    
                    # Update progress
                    param_pbar.update(len(batch))
                    time_pbar.reset()
                
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': self.cv_results
        }
        
    def plot_optimization_results(self, save_path: str = None):
        """Plot optimization results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figure with subplots
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot parameter importance
            param_importance = pd.DataFrame({
                'Parameter': list(self.best_params.keys()),
                'Value': list(self.best_params.values())
            })
            sns.barplot(data=param_importance, x='Parameter', y='Value', ax=ax)
            ax.set_title('Optimal Parameter Values')
            ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
                
        except ImportError:
            print("Matplotlib and seaborn required for plotting")
            
    def save_results(self, path: str):
        """Save optimization results to file."""
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': self.cv_results
        }
        
        import json
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
