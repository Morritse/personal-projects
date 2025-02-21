import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz
from strategy.vwap_obv_strategy import VWAPOBVCrossover
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

def run_with_cached_data():
    """Run strategy using cached data."""
    print("\nRunning with Cached Data:")
    print("-" * 50)
    
    # Load cached data
    cache_file = 'data/market_data.pkl'
    if not os.path.exists(cache_file):
        print("Market data not found")
        return
    
    data = pd.read_pickle(cache_file)
    jnj_df = data['JNJ']
    xlv_df = data['XLV']
    
    # Filter to 2024
    start_lookback = datetime(2023, 12, 1, tzinfo=pytz.UTC)
    start_test = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    end_test = datetime.now(pytz.UTC)
    
    jnj_data = jnj_df[jnj_df.index >= start_lookback].copy()
    xlv_data = xlv_df[xlv_df.index >= start_lookback].copy()
    
    print(f"JNJ Data Range: {jnj_data.index[0]} to {jnj_data.index[-1]}")
    print(f"XLV Data Range: {xlv_data.index[0]} to {xlv_data.index[-1]}")
    print(f"Trading Hours Distribution:")
    print(jnj_data.index.hour.value_counts().sort_index())
    
    # Run strategy
    with open('final_strategy_params.json', 'r') as f:
        config = json.load(f)
    strategy = VWAPOBVCrossover(config)
    trades = strategy.run(jnj_data, xlv_data)
    
    if trades:
        analyze_trades(trades, start_test)

def run_with_fresh_data():
    """Run strategy using fresh Alpaca data."""
    print("\nRunning with Fresh Alpaca Data:")
    print("-" * 50)
    
    # Initialize API client
    load_dotenv()
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    client = StockHistoricalDataClient(api_key, secret_key)
    
    # Set time period
    end_date = datetime.now(pytz.timezone('US/Pacific'))
    start_date = datetime(2023, 12, 1, tzinfo=pytz.timezone('US/Pacific'))
    start_test = datetime(2024, 1, 1, tzinfo=pytz.timezone('US/Pacific'))
    
    # Fetch data
    symbols = ['JNJ', 'XLV']
    data = {}
    
    for symbol in symbols:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Hour,
            start=start_date,
            end=end_date,
            adjustment='raw'
        )
        
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol]
        
        data[symbol] = df
    
    print(f"JNJ Data Range: {data['JNJ'].index[0]} to {data['JNJ'].index[-1]}")
    print(f"XLV Data Range: {data['XLV'].index[0]} to {data['XLV'].index[-1]}")
    print(f"Trading Hours Distribution:")
    print(data['JNJ'].index.hour.value_counts().sort_index())
    
    # Run strategy
    with open('final_strategy_params.json', 'r') as f:
        config = json.load(f)
    strategy = VWAPOBVCrossover(config)
    trades = strategy.run(data['JNJ'], data['XLV'])
    
    if trades:
        analyze_trades(trades, start_test)

def analyze_trades(trades, start_test):
    """Analyze trades and print metrics."""
    trades_df = pd.DataFrame(trades)
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_2024 = trades_df[trades_df['timestamp'] >= start_test]
    
    # Calculate metrics
    sells = trades_2024[trades_2024['action'] == 'SELL']
    total_trades = len(sells)
    winning_trades = len(sells[sells['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    total_pnl = sells['pnl'].sum()
    
    print("\nPerformance Metrics:")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    if total_trades > 0:
        print(f"Average PnL per Trade: ${total_pnl/total_trades:,.2f}")
    
    print("\nTrades by Regime:")
    regime_stats = sells.groupby('regime').agg({
        'pnl': ['count', 'sum', 'mean'],
        'size': 'mean'
    })
    print(regime_stats)
    
    print("\nTrade Distribution by Hour:")
    print(sells['timestamp'].dt.hour.value_counts().sort_index())

if __name__ == "__main__":
    print("Comparing Strategy Implementations")
    print("=" * 50)
    
    run_with_cached_data()
    print("\n")
    run_with_fresh_data()
