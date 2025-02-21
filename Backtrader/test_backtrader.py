import backtrader as bt
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('BacktraderTest')

class TestStrategy(bt.Strategy):
    """Test strategy to verify multi-timeframe data setup."""
    
    def __init__(self):
        """Initialize the strategy with indicators on different timeframes."""
        # Data feeds are available as self.datas[i]
        # self.datas[0] = 1-minute data (primary timeframe)
        # self.datas[1] = 5-minute data
        # self.datas[2] = 15-minute data
        # self.datas[3] = 1-hour data
        # self.datas[4] = daily data
        
        # Example indicators on different timeframes
        self.rsi_1m = bt.indicators.RSI(
            self.datas[0],
            period=14,
            plotname='1min RSI'
        )
        
        self.rsi_5m = bt.indicators.RSI(
            self.datas[1],
            period=14,
            plotname='5min RSI'
        )
        
        self.rsi_15m = bt.indicators.RSI(
            self.datas[2],
            period=14,
            plotname='15min RSI'
        )
        
        self.rsi_1h = bt.indicators.RSI(
            self.datas[3],
            period=14,
            plotname='1h RSI'
        )
        
        self.rsi_1d = bt.indicators.RSI(
            self.datas[4],
            period=14,
            plotname='1d RSI'
        )
    
    def next(self):
        """
        Log data points from different timeframes to verify proper resampling.
        Only log when we have new daily data to avoid excessive output.
        """
        # Check if we have new daily data
        if self.datas[4].datetime.datetime(0) != self.datas[4].datetime.datetime(-1):
            logger.info(
                f'\nTime: {self.datas[0].datetime.datetime(0)}'
                f'\n1min - O: {self.datas[0].open[0]:.2f} H: {self.datas[0].high[0]:.2f} '
                f'L: {self.datas[0].low[0]:.2f} C: {self.datas[0].close[0]:.2f}'
                f'\n5min - O: {self.datas[1].open[0]:.2f} H: {self.datas[1].high[0]:.2f} '
                f'L: {self.datas[1].low[0]:.2f} C: {self.datas[1].close[0]:.2f}'
                f'\n15min - O: {self.datas[2].open[0]:.2f} H: {self.datas[2].high[0]:.2f} '
                f'L: {self.datas[2].low[0]:.2f} C: {self.datas[2].close[0]:.2f}'
                f'\n1hour - O: {self.datas[3].open[0]:.2f} H: {self.datas[3].high[0]:.2f} '
                f'L: {self.datas[3].low[0]:.2f} C: {self.datas[3].close[0]:.2f}'
                f'\nDaily - O: {self.datas[4].open[0]:.2f} H: {self.datas[4].high[0]:.2f} '
                f'L: {self.datas[4].low[0]:.2f} C: {self.datas[4].close[0]:.2f}'
                f'\nRSI Values:'
                f'\n  1min: {self.rsi_1m[0]:.2f}'
                f'\n  5min: {self.rsi_5m[0]:.2f}'
                f'\n  15min: {self.rsi_15m[0]:.2f}'
                f'\n  1hour: {self.rsi_1h[0]:.2f}'
                f'\n  Daily: {self.rsi_1d[0]:.2f}'
            )

def main():
    # Create a cerebro entity
    cerebro = bt.Cerebro()
    
    # Add the test strategy
    cerebro.addstrategy(TestStrategy)
    
    # Load 1-minute data
    data1m = bt.feeds.GenericCSVData(
        dataname='data/historical_data/SPY_1m_20231227_20241226.csv',
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
    
    # Add the 1-minute data feed
    cerebro.adddata(data1m, name='SPY_1m')
    
    # Create and add resampled data feeds
    data5m = cerebro.resampledata(data1m, timeframe=bt.TimeFrame.Minutes, compression=5, name='SPY_5m')
    data15m = cerebro.resampledata(data1m, timeframe=bt.TimeFrame.Minutes, compression=15, name='SPY_15m')
    data1h = cerebro.resampledata(data1m, timeframe=bt.TimeFrame.Minutes, compression=60, name='SPY_1h')
    data1d = cerebro.resampledata(data1m, timeframe=bt.TimeFrame.Days, compression=1, name='SPY_1d')
    
    # Set our desired cash start
    cerebro.broker.setcash(100000.0)
    
    # Print out the starting conditions
    logger.info('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    # Run the strategy
    cerebro.run()
    
    # Print out the final result
    logger.info('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

if __name__ == '__main__':
    main()
