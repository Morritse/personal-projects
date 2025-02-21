from analyze_trades import analyze_trades, compare_window_sizes
from vwap_obv_strategy import VAMEStrategy
from config import config
import pandas as pd
from download_data import load_cached_data, SYMBOLS
from tqdm import tqdm

def run_window_analysis():
    print("Loading data...")
    symbol_data = {}
    for symbol in SYMBOLS:
        df = load_cached_data(symbol)
        if df is not None and not df.empty:
            symbol_data[symbol] = df

    # Test different window sizes
    window_sizes = [8, 12, 16, 20]
    results_by_window = {}
    
    print("\nAnalyzing window sizes...")
    for window in window_sizes:
        print(f"\nTesting window size: {window}")
        all_trades = []
        
        # Update config
        test_config = config.copy()
        test_config['Current Window'] = window
        
        # Run strategy on all symbols
        for symbol, df in tqdm(symbol_data.items(), desc=f"Processing symbols"):
            strategy = VAMEStrategy(test_config)
            trades = strategy.run(df)
            if trades:
                # Add symbol to trades
                for trade in trades:
                    trade['symbol'] = symbol
                all_trades.extend(trades)
        
        # Sort trades by timestamp
        all_trades.sort(key=lambda x: x['timestamp'])
        
        # Analyze combined trades
        results = analyze_trades(all_trades, None, window)
        if results:
            results_by_window[window] = results
    
    # Compare results across window sizes
    compare_window_sizes(results_by_window)

if __name__ == "__main__":
    run_window_analysis()
