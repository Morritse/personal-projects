import backtrader as bt
from strategies.short_term import ShortTermStrategy
import pandas as pd
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger('SignalTest')

def test_symbol(symbol, start_date, end_date):
    """Test strategy on a single symbol and return results."""
    print(f"\n{'='*20} Testing {symbol} {'='*20}")
    
    # Create a cerebro entity
    cerebro = bt.Cerebro()
    
    # Add the strategy
    cerebro.addstrategy(ShortTermStrategy)
    
    # Load 1-minute data
    data = bt.feeds.GenericCSVData(
        dataname=f'data/historical_data/{symbol}_1m_20231227_20241226.csv',
        datetime=0,
        open=1,
        high=2,
        low=3,
        close=4,
        volume=5,
        openinterest=-1,
        dtformat='%Y-%m-%d %H:%M:%S%z',
        timeframe=bt.TimeFrame.Minutes,
        compression=1,
        fromdate=start_date,
        todate=end_date
    )
    
    # Add the data feed
    cerebro.adddata(data)
    
    # Set starting cash
    cerebro.broker.setcash(100000.0)
    
    # Set commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)
    
    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # Run the strategy
    results = cerebro.run()
    strat = results[0]
    
    # Get analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    
    # Print period performance summary
    print(f"\nPeriod Summary:")
    if sharpe and 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
        print(f"Sharpe Ratio: {sharpe['sharperatio']:.3f}")
    else:
        print("Sharpe Ratio: N/A")
        
    if returns and 'rtot' in returns:
        print(f"Total Return: {returns['rtot']*100:.2f}%")
    else:
        print("Total Return: N/A")
    
    if trades.get('total'):
        total_trades = trades['total']['total']
        won_trades = trades['won']['total'] if trades.get('won') else 0
        lost_trades = trades['lost']['total'] if trades.get('lost') else 0
        win_rate = (won_trades / total_trades * 100) if total_trades > 0 else 0
        
        print(f"\nTrade Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        
        if trades.get('won') and trades.get('lost'):
            avg_win = trades['won']['pnl']['average']
            avg_loss = trades['lost']['pnl']['average']
            profit_factor = abs(avg_win/avg_loss) if avg_loss != 0 else 0
            print(f"Average Win: ${avg_win:.2f}")
            print(f"Average Loss: ${avg_loss:.2f}")
            print(f"Profit Factor: {profit_factor:.2f}")
            
    # Return analyzer results
    return {
        'sharpe': sharpe,
        'returns': returns,
        'trades': trades
    }

def main():
    # Test period - Q1 2024
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    # Test on more volatile stocks
    symbols = ['NVDA', 'AMD', 'TSLA']
    
    # Store results for each symbol
    all_results = {}
    total_pnl = 0
    total_trades = 0
    winning_trades = 0
    
    for symbol in symbols:
        results = test_symbol(symbol, start_date, end_date)
        all_results[symbol] = results
        
        # Update aggregates if we have valid results
        if results and 'trades' in results and results['trades'].get('total'):
            total_trades += results['trades']['total']['total']
            if 'won' in results['trades']:
                winning_trades += results['trades']['won']['total']
            if 'pnl' in results['trades']:
                total_pnl += results['trades']['pnl']['net']['total']
    
    # Print aggregate results
    print(f"\n{'='*50}")
    print(f"{'Aggregate Results':^50}")
    print(f"{'='*50}")
    
    print(f"\nOverall Performance:")
    print(f"Total PnL: ${total_pnl:.2f}")
    if total_trades > 0:
        print(f"Total Trades: {total_trades}")
        print(f"Overall Win Rate: {(winning_trades/total_trades*100):.1f}%")
        
        print("\nPer Symbol Breakdown:")
        for symbol, results in all_results.items():
            if results and 'trades' in results and results['trades'].get('total'):
                trades = results['trades']
                won = trades.get('won', {}).get('total', 0)
                total = trades['total']['total']
                win_rate = (won/total*100) if total > 0 else 0
                pnl = trades.get('pnl', {}).get('net', {}).get('total', 0)
                avg_win = trades.get('won', {}).get('pnl', {}).get('average', 0)
                avg_loss = trades.get('lost', {}).get('pnl', {}).get('average', 0)
                profit_factor = abs(avg_win/avg_loss) if avg_loss != 0 else 0
                
                print(f"\n{symbol}:")
                print(f"  PnL: ${pnl:.2f}")
                print(f"  Trades: {total}")
                print(f"  Win Rate: {win_rate:.1f}%")
                print(f"  Avg Win: ${avg_win:.2f}")
                print(f"  Avg Loss: ${avg_loss:.2f}")
                print(f"  Profit Factor: {profit_factor:.2f}")

if __name__ == '__main__':
    main()
