import backtrader as bt
import pandas as pd
import datetime as dt
import pytz
from tqdm import tqdm
from ensemble import EnsembleStrategy

def run_symbol(symbol, csv_file):
    """Run ensemble strategy on a single symbol."""
    print(f"\nProcessing {symbol} from {csv_file}")
    
    # Create Cerebro
    cerebro = bt.Cerebro()
    
    try:
        # Load and validate data first
        df = pd.read_csv(csv_file)
        total_bars = len(df)
        print(f"Loaded {total_bars} bars from CSV")
        print("\nSample data:")
        print(df.head())
        print("\nColumns:", df.columns.tolist())
        
        # Create progress bar with known total
        progress_bar = tqdm(
            total=total_bars,
            desc=f"Processing {symbol}",
            unit=" bars",
            dynamic_ncols=True
        )
        
        # Load 1-minute data
        print("\nLoading 1-minute data...")
        data1m = bt.feeds.GenericCSVData(
            dataname=csv_file,
            dtformat='%Y-%m-%d %H:%M:%S%z',  # UTC format with timezone
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
            tz=pytz.timezone('US/Eastern'),
            debug=True
        )
        
        # Add data feeds
        cerebro.adddata(data1m, name='1m')
        print("1-min data added")
        
        # Resample to 5-min
        print("Resampling to 5-min...")
        data5m = cerebro.resampledata(
            data1m,
            timeframe=bt.TimeFrame.Minutes,
            compression=5,
            name='5m'
        )
        print("5-min data added")
        
        # Resample to 15-min
        print("Resampling to 15-min...")
        data15m = cerebro.resampledata(
            data1m,
            timeframe=bt.TimeFrame.Minutes,
            compression=15,
            name='15m'
        )
        print("15-min data added")
        
        # Add strategy with progress bar
        cerebro.addstrategy(EnsembleStrategy, progress_bar=progress_bar)
        
        # Run
        print(f"\nStarting backtest for {symbol}...")
        cerebro.run()
        
        # Close progress bar
        progress_bar.close()
        print(f"Backtest complete for {symbol}")
        
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.close()

def run_test():
    """Run test on multiple symbols."""
    symbols = {
        'AAPL': 'data/historical_data/AAPL_1m_20231227_20241226.csv',
        'MSFT': 'data/historical_data/MSFT_1m_20231227_20241226.csv',
        'NVDA': 'data/historical_data/NVDA_1m_20231227_20241226.csv',
        'META': 'data/historical_data/META_1m_20231227_20241226.csv',
        'GOOGL': 'data/historical_data/GOOGL_1m_20231227_20241226.csv'
    }
    
    # Start with just one symbol for testing
    symbol = 'AAPL'
    csv_file = symbols[symbol]
    run_symbol(symbol, csv_file)

if __name__ == '__main__':
    run_test()
