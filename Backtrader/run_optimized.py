import backtrader as bt
import pandas as pd
import datetime as dt
import pytz
from ensemble_optimized import EnsembleStrategy

def run_symbol(symbol, csv_file):
    """Run ensemble strategy on a single symbol."""
    print(f"\nStarting analysis for {symbol}...")
    
    # Create Cerebro
    cerebro = bt.Cerebro(stdstats=False)  # Disable default observers
    
    try:
        # Load 1-minute data
        data1m = bt.feeds.GenericCSVData(
            dataname=csv_file,
            dtformat='%Y-%m-%d %H:%M:%S%z',
            datetime=0,      # Column index for timestamp
            open=1,          # Column index for open
            high=2,          # Column index for high
            low=3,           # Column index for low
            close=4,         # Column index for close
            volume=5,        # Column index for volume
            openinterest=-1,
            headers=True,    # CSV has headers
            timeframe=bt.TimeFrame.Minutes,
            compression=1,
            sessionstart=dt.time(9, 30),
            sessionend=dt.time(16, 0),
            tz=pytz.timezone('US/Eastern')
        )
        
        # Add data feeds
        cerebro.adddata(data1m, name=symbol)
        
        # Resample to 5-min
        data5m = cerebro.resampledata(
            data1m,
            timeframe=bt.TimeFrame.Minutes,
            compression=5,
            name=f"{symbol}_5m"
        )
        
        # Resample to 15-min
        data15m = cerebro.resampledata(
            data1m,
            timeframe=bt.TimeFrame.Minutes,
            compression=15,
            name=f"{symbol}_15m"
        )
        
        # Add strategy
        cerebro.addstrategy(EnsembleStrategy)
        
        # Run
        cerebro.run()
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")

def run_test():
    """Run test on a single symbol."""
    # Test with AAPL
    symbol = 'AAPL'
    csv_file = 'data/historical_data/AAPL_1m_20231227_20241226.csv'
    
    print(f"\nTesting optimized ensemble strategy on {symbol}")
    run_symbol(symbol, csv_file)

if __name__ == '__main__':
    run_test()
