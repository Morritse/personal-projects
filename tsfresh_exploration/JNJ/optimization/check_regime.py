import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from strategy.vwap_obv_strategy import VWAPOBVCrossover
import pytz
from zoneinfo import ZoneInfo

# Load environment variables
load_dotenv()

def check_regime_for_date(target_date):
    # Initialize API client
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    client = StockHistoricalDataClient(api_key, secret_key)
    
    # Load strategy config
    with open('final_strategy_params.json', 'r') as f:
        config = json.load(f)
    
    # Initialize strategy
    strategy = VWAPOBVCrossover(config)
    
    # Calculate time range (30 days before target date)
    end = target_date + timedelta(days=1)  # Include full target date
    start = target_date - timedelta(days=30)
    
    print(f"\nAnalyzing regime for: {target_date.strftime('%Y-%m-%d')}")
    print(f"Using data from: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    # Fetch XLV data
    request = StockBarsRequest(
        symbol_or_symbols='XLV',
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
        adjustment='raw'
    )
    
    try:
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc['XLV']
        
        # Filter to target date
        target_data = df[df.index.date == target_date.date()]
        
        if target_data.empty:
            print(f"No data available for {target_date.strftime('%Y-%m-%d')}")
            return
        
        # Calculate regime
        returns = df['close'].pct_change()
        ret = returns.rolling(window=strategy.regime_window).mean() * 252
        vol = returns.rolling(window=strategy.regime_window).std() * np.sqrt(252)
        
        # Get values for target date
        target_ret = ret[ret.index.date == target_date.date()]
        target_vol = vol[vol.index.date == target_date.date()]
        vol_67pct = vol.quantile(0.67)
        
        print("\nXLV Regime Analysis:")
        print(f"Date: {target_date.strftime('%Y-%m-%d')}")
        
        # Show hourly analysis
        print("\nHourly Analysis:")
        for hour in target_ret.index:
            current_ret = target_ret[hour]
            current_vol = target_vol[hour]
            
            # Determine regime
            if current_ret > 0 and current_vol > vol_67pct:
                regime = 'bull_high_vol'
            elif current_ret <= 0 and current_vol > vol_67pct:
                regime = 'bear_high_vol'
            else:
                regime = 'not_high_vol'
            
            print(f"\nTime: {hour.strftime('%H:%M')}")
            print(f"Return: {current_ret:.2%}")
            print(f"Volatility: {current_vol:.2%}")
            print(f"67th Percentile Vol: {vol_67pct:.2%}")
            print(f"High Vol?: {'Yes' if current_vol > vol_67pct else 'No'}")
            print(f"Regime: {regime}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check December 30th, 2024
    target_date = datetime(2024, 12, 30, tzinfo=ZoneInfo('America/Los_Angeles'))
    check_regime_for_date(target_date)
