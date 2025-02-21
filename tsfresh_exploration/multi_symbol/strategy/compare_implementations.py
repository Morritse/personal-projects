import os
import json
import time
import pandas as pd
from datetime import datetime
from vwap_obv_strategy import VWAPOBVCrossover
from vwap_obv_strategy_vec import VWAPOBVCrossoverVec

def load_data(symbol: str):
    """Load market data for a symbol."""
    cache_file = f'../data/cache/{symbol.lower()}_data.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Market data not found for {symbol}")
    
    data = pd.read_pickle(cache_file)
    
    # Convert UTC timestamps to naive timestamps
    data.index = data.index.tz_localize(None)
    
    # Filter to 2022-2024
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    return data

def compare_trades(trades1, trades2):
    """Compare trades from both implementations."""
    df1 = pd.DataFrame(trades1)
    df2 = pd.DataFrame(trades2)
    
    print("\nTrade Comparison:")
    print(f"Original Implementation: {len(trades1)} trades")
    print(f"Vectorized Implementation: {len(trades2)} trades")
    
    if not trades1 or not trades2:
        return
    
    # Compare entry/exit points
    entries1 = df1[df1['action'] == 'BUY']['timestamp'].tolist()
    entries2 = df2[df2['action'] == 'BUY']['timestamp'].tolist()
    
    exits1 = df1[df1['action'] == 'SELL']['timestamp'].tolist()
    exits2 = df2[df2['action'] == 'SELL']['timestamp'].tolist()
    
    entry_matches = sum(1 for e in entries1 if e in entries2)
    exit_matches = sum(1 for e in exits1 if e in exits2)
    
    print(f"\nEntry Points Match: {entry_matches}/{len(entries1)} ({entry_matches/len(entries1):.1%})")
    print(f"Exit Points Match: {exit_matches}/{len(exits1)} ({exit_matches/len(exits1):.1%})")
    
    # Compare PnL
    pnl1 = df1[df1['action'] == 'SELL']['pnl'].sum()
    pnl2 = df2[df2['action'] == 'SELL']['pnl'].sum()
    
    print(f"\nOriginal PnL: ${pnl1:,.2f}")
    print(f"Vectorized PnL: ${pnl2:,.2f}")
    print(f"Difference: ${pnl2-pnl1:,.2f} ({(pnl2-pnl1)/abs(pnl1):.1%})")

def test_implementations():
    """Test both implementations on multiple symbols."""
    # Test parameters
    params = {
        'vwap_length': 50,
        'mfi_length': 9,
        'mfi_oversold': 30,
        'mfi_overbought': 70,
        'regime_window': 20,
        'vol_percentile': 67,
        'size_factor': 1.0,
        'max_position': 0.25,
        'max_hold_hours': 24,
        'profit_target': 0.02,
        'stop_loss': 0.02
    }
    
    # Test symbols
    symbols = ['META', 'NVDA', 'TSLA', 'AAPL', 'AMD']
    
    results = []
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        print("=" * 50)
        
        try:
            # Load data
            data = load_data(symbol)
            
            # Test original implementation
            start_time = time.time()
            strategy1 = VWAPOBVCrossover(params)
            trades1 = strategy1.run(data)
            time1 = time.time() - start_time
            
            # Test vectorized implementation
            start_time = time.time()
            strategy2 = VWAPOBVCrossoverVec(params)
            trades2 = strategy2.run(data)
            time2 = time.time() - start_time
            
            # Compare results
            print(f"\nPerformance:")
            print(f"Original Implementation: {time1:.2f} seconds")
            print(f"Vectorized Implementation: {time2:.2f} seconds")
            print(f"Speedup: {time1/time2:.1f}x")
            
            compare_trades(trades1, trades2)
            
            # Store results
            results.append({
                'symbol': symbol,
                'orig_time': time1,
                'vec_time': time2,
                'speedup': time1/time2,
                'orig_trades': len(trades1),
                'vec_trades': len(trades2),
                'orig_pnl': sum(t['pnl'] for t in trades1 if t['action'] == 'SELL'),
                'vec_pnl': sum(t['pnl'] for t in trades2 if t['action'] == 'SELL')
            })
            
        except Exception as e:
            print(f"Error testing {symbol}: {e}")
    
    # Save summary
    if results:
        df = pd.DataFrame(results)
        print("\nOverall Results:")
        print("=" * 50)
        print(f"\nAverage Speedup: {df['speedup'].mean():.1f}x")
        print(f"Average PnL Difference: ${(df['vec_pnl'] - df['orig_pnl']).mean():,.2f}")
        print(f"\nDetailed Results:")
        print(df)
        
        # Save to file
        os.makedirs('results', exist_ok=True)
        df.to_csv('results/implementation_comparison.csv')
        print("\nResults saved to results/implementation_comparison.csv")

if __name__ == "__main__":
    test_implementations()
