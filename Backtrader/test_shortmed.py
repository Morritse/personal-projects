import backtrader as bt
from shortmed import EnsembleStrategy
import os

def run_ensemble_analysis(csv_path):
    """
    Run the ensemble strategy on a single symbol.
    """
    # Create a Cerebro engine
    cerebro = bt.Cerebro()

    # Load 1-minute data
    data1m = bt.feeds.GenericCSVData(
        dataname=csv_path,
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        dtformat='%Y-%m-%d %H:%M:%S%z',
        timeframe=bt.TimeFrame.Minutes,
        compression=1
    )

    # Add data feeds in order:
    # 1. Original 1-min data
    cerebro.adddata(data1m, name='1m')
    
    # 2. Resample to 5-min
    data5m = cerebro.resampledata(
        data1m,
        timeframe=bt.TimeFrame.Minutes,
        compression=5,
        name='5m'
    )

    # Add ensemble strategy
    cerebro.addstrategy(EnsembleStrategy)

    # Run analysis
    cerebro.run()

if __name__ == '__main__':
    """Analyze all symbols with ensemble strategy."""
    # Get all CSV files from historical_data directory
    data_dir = 'data/historical_data'
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_1m_20231227_20241226.csv')]
    symbols = sorted([f.split('_')[0] for f in csv_files])
    
    print(f"Found {len(symbols)} symbols to analyze:")
    print(", ".join(symbols))
    print("\nStarting ensemble analysis...")
    
    # Run analysis for each symbol
    for symbol in symbols:
        print(f"\n{'='*20} Analyzing {symbol} {'='*20}")
        csv_file = f'{data_dir}/{symbol}_1m_20231227_20241226.csv'
        run_ensemble_analysis(csv_file)
