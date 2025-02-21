import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from strategy.vwap_obv_strategy import VWAPOBVCrossover
import pytz
from run_strategy import run_strategy
from multi_symbol_backtest import MultiSymbolBacktest

def compare_strategies():
    """Compare results between run_strategy and multi_symbol_backtest."""
    # Set up test period - use 3 years from now
    pst = pytz.timezone('US/Pacific')
    end_date = datetime.now(pst)
    start_date = end_date - timedelta(days=365*3)  # 3 years
    
    print(f"Testing period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    print("\nRunning original strategy...")
    original_trades, original_metrics = run_strategy()
    
    print("\nRunning multi-symbol backtest...")
    backtest = MultiSymbolBacktest(['JNJ'], start_date, end_date)
    backtest_results = backtest.run_backtest()
    
    if original_trades is None or backtest_results is None:
        print("One or both strategies produced no trades")
        return
    
    # Convert backtest trades to comparable format
    backtest_trades = pd.DataFrame(backtest_results['trades']['JNJ'])
    
    # Compare trade counts
    print("\nTrade Count Comparison:")
    print(f"Original Strategy: {len(original_trades)}")
    print(f"Multi-Symbol Backtest: {len(backtest_trades)}")
    
    # Compare trade dates
    original_dates = set(original_trades['timestamp'])
    backtest_dates = set(backtest_trades['entry_time'])
    
    common_dates = original_dates.intersection(backtest_dates)
    only_original = original_dates - backtest_dates
    only_backtest = backtest_dates - original_dates
    
    print("\nTrade Date Analysis:")
    print(f"Common Trade Dates: {len(common_dates)}")
    print(f"Only in Original: {len(only_original)}")
    print(f"Only in Backtest: {len(only_backtest)}")
    
    if only_original:
        print("\nDates only in Original Strategy:")
        for date in sorted(list(only_original))[:5]:  # Show first 5
            print(date)
    
    if only_backtest:
        print("\nDates only in Backtest:")
        for date in sorted(list(only_backtest))[:5]:  # Show first 5
            print(date)
    
    # Compare position sizes for common dates
    if common_dates:
        print("\nPosition Size Analysis for Common Dates:")
        for date in sorted(list(common_dates))[:5]:  # Show first 5
            orig_trade = original_trades[original_trades['timestamp'] == date].iloc[0]
            back_trade = backtest_trades[backtest_trades['entry_time'] == date].iloc[0]
            
            print(f"\nDate: {date}")
            print(f"Original Size: {orig_trade['size']}")
            print(f"Backtest Size: {back_trade['shares']}")
            print(f"Original PnL: ${orig_trade['pnl']:,.2f}")
            print(f"Backtest PnL: ${back_trade['pnl']:,.2f}")
    
    # Compare overall metrics
    print("\nOverall Metrics Comparison:")
    print("Original Strategy:")
    print(f"Win Rate: {original_metrics['win_rate']:.1%}")
    print(f"Total PnL: ${original_metrics['total_pnl']:,.2f}")
    print(f"Max Drawdown: {original_metrics['max_drawdown']:.1%}")
    
    print("\nMulti-Symbol Backtest:")
    print(f"Win Rate: {backtest_results['win_rate']:.1%}")
    print(f"Total PnL: ${backtest_results['total_pnl']:,.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.1%}")

if __name__ == "__main__":
    compare_strategies()
