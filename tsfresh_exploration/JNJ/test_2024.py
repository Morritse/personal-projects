import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz
from strategy.vwap_obv_strategy import VWAPOBVCrossover

def test_2024_only():
    """Test strategy on 2024 data only."""
    # Load data
    cache_file = 'data/market_data.pkl'
    if not os.path.exists(cache_file):
        print("Market data not found")
        return
    
    data = pd.read_pickle(cache_file)
    jnj_df = data['JNJ']
    xlv_df = data['XLV']
    
    # Convert index to tz-naive
    jnj_df.index = jnj_df.index.tz_localize(None)
    xlv_df.index = xlv_df.index.tz_localize(None)
    
    # Filter to 2024 only (include December 2023 for lookback)
    start_lookback = datetime(2023, 12, 1)
    start_test = datetime(2024, 1, 1)
    end_test = datetime(2024, 1, 31)  # Just test January first
    
    # Include lookback period for indicators
    jnj_data = jnj_df[(jnj_df.index >= start_lookback) & (jnj_df.index <= end_test)].copy()
    xlv_data = xlv_df[(xlv_df.index >= start_lookback) & (xlv_df.index <= end_test)].copy()
    
    print(f"Data ranges:")
    print(f"JNJ: {jnj_data.index[0]} to {jnj_data.index[-1]}")
    print(f"XLV: {xlv_data.index[0]} to {xlv_data.index[-1]}")
    
    print("\nData points:")
    print(f"JNJ: {len(jnj_data)} bars")
    print(f"XLV: {len(xlv_data)} bars")
    
    # Load strategy
    with open('final_strategy_params.json', 'r') as f:
        config = json.load(f)
    strategy = VWAPOBVCrossover(config)
    
    # Debug: Check regime classification
    print("\nRegime Analysis:")
    print("-" * 50)
    
    # Calculate returns and volatility
    returns = xlv_data['close'].pct_change()
    ret = returns.rolling(window=strategy.regime_window).mean() * 252
    vol = returns.rolling(window=strategy.regime_window).std() * np.sqrt(252)
    
    # Get volatility threshold
    vol_67pct = vol.quantile(0.67)
    print(f"Volatility 67th percentile: {vol_67pct:.2%}")
    
    # Print last few days of metrics
    print("\nLast 5 days of metrics:")
    for i in range(-5, 0):
        date = xlv_data.index[i]
        curr_ret = ret.iloc[i]
        curr_vol = vol.iloc[i]
        regime = "bull_high_vol" if curr_ret > 0 and curr_vol > vol_67pct else \
                "bear_high_vol" if curr_ret <= 0 and curr_vol > vol_67pct else "none"
        
        print(f"\n{date.strftime('%Y-%m-%d %H:%M')}")
        print(f"Return (annualized): {curr_ret:.2%}")
        print(f"Volatility (annualized): {curr_vol:.2%}")
        print(f"Regime: {regime}")
    
    # Run strategy
    trades = strategy.run(jnj_data, xlv_data)
    
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Filter to 2024 trades only
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_2024 = trades_df[trades_df['timestamp'] >= start_test]
        
        # Calculate metrics
        sells = trades_2024[trades_2024['action'] == 'SELL']
        total_trades = len(sells)
        winning_trades = len(sells[sells['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        total_pnl = sells['pnl'].sum()
        
        print("\n2024 Performance:")
        print("-" * 50)
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
        
        print("\nDetailed Trades:")
        for _, trade in sells.iterrows():
            print(f"\n{trade['timestamp'].strftime('%Y-%m-%d %H:%M')}")
            print(f"Regime: {trade['regime']}")
            print(f"Size: {trade['size']}")
            print(f"PnL: ${trade['pnl']:,.2f}")
            print(f"Reason: {trade.get('reason', 'unknown')}")
    else:
        print("\nNo trades generated")

if __name__ == "__main__":
    test_2024_only()
