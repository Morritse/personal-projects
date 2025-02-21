import pandas as pd
import numpy as np
from datetime import datetime

def calculate_buy_hold_return(symbol: str):
    """Calculate buy and hold return for a symbol."""
    # Load data
    data = pd.read_pickle(f'../data/cache/{symbol.lower()}_data.pkl')
    
    # Convert UTC timestamps to naive timestamps
    data.index = data.index.tz_localize(None)
    
    # Filter to 2022-2024
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    # Calculate returns
    start_price = data['close'].iloc[0]
    end_price = data['close'].iloc[-1]
    total_return = (end_price - start_price) / start_price
    
    # Calculate holding period in years
    days_held = (data.index[-1] - data.index[0]).days
    years_held = days_held / 365
    
    # Calculate annualized return
    annual_return = (1 + total_return) ** (1/years_held) - 1
    
    print(f"\nBuy & Hold Analysis for {symbol}:")
    print(f"Start Date: {data.index[0]}")
    print(f"End Date: {data.index[-1]}")
    print(f"Start Price: ${start_price:.2f}")
    print(f"End Price: ${end_price:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annual_return:.2%}")
    
    # Calculate drawdown
    rolling_max = data['close'].expanding().max()
    drawdowns = (data['close'] - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()
    print(f"Max Drawdown: {max_drawdown:.2%}")
    
    return {
        'symbol': symbol,
        'start_date': data.index[0],
        'end_date': data.index[-1],
        'start_price': start_price,
        'end_price': end_price,
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown
    }

if __name__ == "__main__":
    # Calculate for all symbols
    symbols = ['META', 'NVDA', 'TSLA', 'AAPL', 'AMD', 'COIN', 'JNJ']
    results = []
    
    for symbol in symbols:
        try:
            result = calculate_buy_hold_return(symbol)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)
        df = df.set_index('symbol')
        df['total_return'] = df['total_return'].map('{:.2%}'.format)
        df['annual_return'] = df['annual_return'].map('{:.2%}'.format)
        df['max_drawdown'] = df['max_drawdown'].map('{:.2%}'.format)
        print("\nSummary of Buy & Hold Returns:")
        print(df)
