import multiprocessing as mp
import backtrader as bt
import os
import pandas as pd
from tqdm import tqdm
import datetime as dt
import pytz
from ensemble_optimized import EnsembleStrategy

def run_symbol(csv_file):
    """Run ensemble strategy on a single symbol."""
    try:
        # Get symbol from filename
        symbol = os.path.basename(csv_file).split('_')[0]
        
        # Get total bars for progress
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                raise ValueError("Empty dataframe")
            if df.isnull().values.any():
                print("Warning: CSV contains null values")
            if (df[['close', 'volume']] <= 0).any().any():
                print("Warning: CSV contains zero/negative prices or volumes")
            total_bars = len(df)
            print(f"Loaded {total_bars} bars from {csv_file}")
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            return
        
        # Create progress bar
        progress_bar = tqdm(
            total=total_bars,
            desc=f"Analyzing {symbol}",
            unit=" bars",
            position=0,
            leave=True
        )
        
        # Create Cerebro with debug output
        cerebro = bt.Cerebro(stdstats=False)  # Disable default observers
        print("\nCreating Cerebro instance...")
        
        # Load and validate 1-minute data
        print("Loading 1-minute data...")
        
        # Verify data format
        sample = pd.read_csv(csv_file, nrows=5)
        print("\nSample data:")
        print(sample.head())
        print("\nColumns:", sample.columns.tolist())
        
        # Load data directly
        print("\nLoading data with debug output...")
        data1m = bt.feeds.GenericCSVData(
            dataname=csv_file,
            dtformat='%Y-%m-%d %H:%M:%S%z',  # Match exact format from sample
            headers=True,    # CSV has headers
            datetime=0,      # Use column indices after header
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,
            timeframe=bt.TimeFrame.Minutes,
            compression=1,
            sessionstart=dt.time(9, 30),  # Market hours in ET
            sessionend=dt.time(16, 0),
            tz=pytz.timezone('US/Eastern'),
            debug=True
        )
        
        # Load data into Cerebro
        print("Loading data into Cerebro...")
        cerebro.adddata(data1m, name='1m')
        
        # Run preprocessing to load data
        print("Running preprocessing...")
        cerebro.runstop()  # This loads the data
        
        # Print first few lines if data was loaded
        if len(data1m) > 0:
            print("\nFirst few lines of parsed data:")
            for i in range(min(5, len(data1m))):
                dt = bt.num2date(data1m.lines.datetime[i])
                print(f"Bar {i}: {dt}, Open: {data1m.lines.open[i]}, Close: {data1m.lines.close[i]}")
        else:
            print("\nWARNING: No data was loaded!")
            return
            
        print(f"Loaded {len(data1m)} 1-minute bars")
        
        # Resample with validation
        print("\nResampling to 5-min...")
        data5m = cerebro.resampledata(
            data1m,
            timeframe=bt.TimeFrame.Minutes,
            compression=5,
            name='5m'
        )
        print("5-min resampling complete")
        
        print("\nResampling to 15-min...")
        data15m = cerebro.resampledata(
            data1m,
            timeframe=bt.TimeFrame.Minutes,
            compression=15,
            name='15m'
        )
        print("15-min resampling complete")
        
        # Add strategy with progress bar
        print("\nAdding strategy...")
        cerebro.addstrategy(EnsembleStrategy, progress_bar=progress_bar)
        
        # Run strategy
        print("\nRunning strategy...")
        cerebro.run()
        
        # Close progress bar
        progress_bar.close()
        
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        if 'progress_bar' in locals():
            progress_bar.close()

def run_in_parallel(symbol_csv_files):
    """Run analysis on multiple symbols in parallel."""
    print(f"Running ensemble analysis on {len(symbol_csv_files)} symbols using {mp.cpu_count()} processes...")
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(run_symbol, symbol_csv_files)

if __name__ == '__main__':
    # Test with AAPL
    symbol = 'AAPL'
    csv_file = 'data/historical_data/AAPL_1m_20231227_20241226.csv'
    run_symbol(csv_file)
