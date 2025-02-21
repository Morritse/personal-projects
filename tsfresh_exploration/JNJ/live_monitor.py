import os
import json
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from strategy.vwap_obv_strategy import VWAPOBVCrossover
from time import sleep
import pytz

# Load environment variables
load_dotenv()

class PositionManager:
    def __init__(self):
        self.active_position = None
        self.position_history = []
    
    def enter_position(self, entry_price, size, stop_loss, take_profit, timestamp, regime, mfi_exit):
        """Record new position entry."""
        if self.active_position:
            return False  # Already in a position
        
        self.active_position = {
            'entry_price': entry_price,
            'size': size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': timestamp,
            'regime': regime,
            'highest_price': entry_price,  # For trailing stop
            'mfi_exit': mfi_exit  # Store regime-specific MFI exit level
        }
        return True
    
    def update_position(self, current_price, timestamp, vwap, mfi):
        """Update position status and check for exit conditions."""
        if not self.active_position:
            return None
        
        # Calculate holding time
        entry_time = self.active_position['entry_time']
        hours_held = (timestamp - entry_time).total_seconds() / 3600
        
        # Update trailing stop if price higher
        self.active_position['highest_price'] = max(
            self.active_position['highest_price'],
            current_price
        )
        
        # Calculate trailing stop
        trail_distance = (self.active_position['take_profit'] - self.active_position['entry_price']) * 0.5
        trailing_stop = self.active_position['highest_price'] - trail_distance
        stop_loss = max(self.active_position['stop_loss'], trailing_stop)
        
        # Check exit conditions
        exit_reason = None
        if current_price <= stop_loss:
            exit_reason = 'stop_loss'
        elif current_price >= self.active_position['take_profit']:
            exit_reason = 'take_profit'
        elif hours_held >= 36:  # Max hold time
            exit_reason = 'max_hold_time'
        # Technical exits (same as backtest)
        elif current_price > vwap:
            exit_reason = 'vwap_cross'
        elif mfi > self.active_position['mfi_exit']:
            exit_reason = 'mfi_overbought'
        
        if exit_reason:
            pnl = (current_price - self.active_position['entry_price']) * self.active_position['size']
            exit_data = {
                **self.active_position,
                'exit_price': current_price,
                'exit_time': timestamp,
                'pnl': pnl,
                'exit_reason': exit_reason
            }
            self.position_history.append(exit_data)
            self.active_position = None
            return exit_data
        
        return None

def load_config():
    """Load strategy configuration."""
    with open('final_strategy_params.json', 'r') as f:
        return json.load(f)

def fetch_live_data(symbol, lookback_days=30):
    """Fetch recent hourly data from Alpaca."""
    # Load API credentials from .env
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        raise ValueError("Alpaca API credentials not found in .env file")
    
    client = StockHistoricalDataClient(api_key, secret_key)
    
    # Calculate time range in UTC
    pst = pytz.timezone('US/Pacific')
    utc = pytz.UTC
    
    end = datetime.now(pst).astimezone(utc)
    start = end - timedelta(days=lookback_days)
    
    # Request hourly bars
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
        adjustment='raw'  # Get raw data
    )
    
    try:
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if len(df) == 0:
            raise ValueError(f"No data returned for {symbol}")
        
        # Handle multi-symbol response
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol]
        
        # Convert UTC to PST
        df.index = df.index.tz_convert(pst)
        
        # Prepare data format
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        print(f"\nLatest data for {symbol}:")
        print(f"Timestamp (PST): {df.index[-1]}")
        print(f"Bars: {len(df)}")
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        raise

def calculate_indicators(jnj_data, xlv_data, config):
    """Calculate all strategy indicators."""
    strategy = VWAPOBVCrossover(config)
    
    # Get latest data point
    current_jnj = jnj_data.iloc[-1]
    price = current_jnj['close']
    
    # Calculate VWAP
    vwap = strategy.calculate_vwap(jnj_data, strategy.vwap_window).iloc[-1]
    
    # Calculate MFI
    mfi = strategy.calculate_mfi(jnj_data).iloc[-1]
    
    # Calculate OBV
    obv = strategy.calculate_obv(jnj_data)
    obv_change = obv.diff().iloc[-1]
    
    # Calculate regime
    regime = strategy.classify_regime(xlv_data)
    
    # Check conditions
    price_below_vwap = price < vwap
    obv_falling = obv_change < 0
    mfi_oversold = mfi < strategy.mfi_entry
    
    # Get regime parameters if in high vol regime
    regime_params = strategy.get_regime_parameters(regime) if regime else None
    
    # Calculate potential position size if entry conditions met
    atr = strategy.calculate_atr(jnj_data).iloc[-1]
    position_size = None
    stop_loss = None
    take_profit = None
    
    if all([price_below_vwap, obv_falling, mfi_oversold, regime]):
        params = regime_params
        raw_stop = params['stop_mult'] * atr
        stop_distance = min(max(raw_stop, strategy.min_stop_dollars), strategy.max_stop_dollars)
        
        risk_amount = 100000 * strategy.risk_per_trade  # Using $100k capital
        risk_per_share = atr * params['stop_mult']
        base_size = risk_amount / risk_per_share
        position_size = round(base_size * params['position_scale'])
        
        stop_loss = price - stop_distance
        take_profit = price + (stop_distance * params['reward_risk'])
    
    return {
        'timestamp': jnj_data.index[-1],
        'price': price,
        'vwap': vwap,
        'mfi': mfi,
        'obv_change': obv_change,
        'regime': regime,
        'conditions': {
            'price_below_vwap': price_below_vwap,
            'obv_falling': obv_falling,
            'mfi_oversold': mfi_oversold,
            'high_vol_regime': regime is not None
        },
        'regime_params': regime_params,
        'position_details': {
            'size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        } if position_size else None
    }

def format_output(indicators, position_manager):
    """Format indicator data for display."""
    print("\n" + "="*50)
    print(f"Time (PST): {indicators['timestamp']}")
    print("-"*50)
    
    # Show active position if any
    if position_manager.active_position:
        pos = position_manager.active_position
        hours_held = (indicators['timestamp'] - pos['entry_time']).total_seconds() / 3600
        unrealized_pnl = (indicators['price'] - pos['entry_price']) * pos['size']
        
        print("\nActive Position:")
        print(f"Entry Price: ${pos['entry_price']:.2f}")
        print(f"Current Price: ${indicators['price']:.2f}")
        print(f"Size: {pos['size']} shares")
        print(f"Stop Loss: ${pos['stop_loss']:.2f}")
        print(f"Take Profit: ${pos['take_profit']:.2f}")
        print(f"Hours Held: {hours_held:.1f}")
        print(f"Unrealized PnL: ${unrealized_pnl:.2f}")
        print(f"Regime: {pos['regime']}")
        print(f"MFI Exit: {pos['mfi_exit']}")
    
    print("\nPrice Action:")
    print(f"Current Price: ${indicators['price']:.2f}")
    print(f"VWAP (50): ${indicators['vwap']:.2f}")
    print(f"Price vs VWAP: {'BELOW' if indicators['conditions']['price_below_vwap'] else 'ABOVE'}")
    
    print("\nMomentum:")
    print(f"MFI (9): {indicators['mfi']:.1f}")
    print(f"OBV Change: {indicators['obv_change']:.0f}")
    
    print("\nRegime:")
    print(f"Current: {indicators['regime'] if indicators['regime'] else 'Not High Vol'}")
    
    if indicators['regime_params']:
        print("\nRegime Parameters:")
        for key, value in indicators['regime_params'].items():
            print(f"  {key}: {value}")
    
    print("\nSignal Conditions:")
    conditions = indicators['conditions']
    print(f"✓ Price Below VWAP: {conditions['price_below_vwap']}")
    print(f"✓ OBV Falling: {conditions['obv_falling']}")
    print(f"✓ MFI Oversold: {conditions['mfi_oversold']}")
    print(f"✓ High Vol Regime: {conditions['high_vol_regime']}")
    
    # Entry signal
    entry_signal = all([
        conditions['price_below_vwap'],
        conditions['obv_falling'],
        conditions['mfi_oversold'],
        conditions['high_vol_regime']
    ])
    
    # Only show entry signal if no active position
    if not position_manager.active_position:
        print("\nSignal:", "🟢 ENTRY" if entry_signal else "⚪ WAIT")
        
        if entry_signal and indicators['position_details']:
            details = indicators['position_details']
            print("\nPotential Position:")
            print(f"Size: {details['size']} shares")
            print(f"Stop Loss: ${details['stop_loss']:.2f}")
            print(f"Take Profit: ${details['take_profit']:.2f}")
    
    print("="*50 + "\n")

def monitor_live():
    """Main monitoring loop."""
    config = load_config()
    update_interval = 300  # 5 minutes
    pst = pytz.timezone('US/Pacific')
    position_manager = PositionManager()
    
    print("Starting live monitor...")
    print("Press Ctrl+C to exit")
    print(f"Updating every {update_interval//60} minutes")
    print(f"Current time (PST): {datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    try:
        while True:
            now = datetime.now(pst).time()
            
            # Only fetch during market hours (PST)
            if now < time(6, 30) or now > time(13, 0):  # 6:30 AM - 1:00 PM PST
                print("\nMarket closed. Waiting for next session...")
                # Sleep until next market open
                next_open = datetime.now(pst)
                if now > time(13, 0):  # After close
                    next_open += timedelta(days=1)
                next_open = next_open.replace(hour=6, minute=30, second=0)
                sleep_seconds = (next_open - datetime.now(pst)).total_seconds()
                sleep(sleep_seconds)
                continue
            
            # Fetch latest data
            print("\nFetching latest data...")
            jnj_data = fetch_live_data('JNJ')
            xlv_data = fetch_live_data('XLV')
            
            # Calculate indicators
            indicators = calculate_indicators(jnj_data, xlv_data, config)
            
            # Check for position exit
            if position_manager.active_position:
                exit_data = position_manager.update_position(
                    indicators['price'],
                    indicators['timestamp'],
                    indicators['vwap'],
                    indicators['mfi']
                )
                if exit_data:
                    print("\nPosition Exit:")
                    print(f"Exit Price: ${exit_data['exit_price']:.2f}")
                    print(f"PnL: ${exit_data['pnl']:.2f}")
                    print(f"Reason: {exit_data['exit_reason']}")
            
            # Check for position entry
            elif all(indicators['conditions'].values()):
                details = indicators['position_details']
                if details:
                    # Get regime-specific MFI exit level
                    mfi_exit = (
                        indicators['regime_params']['mfi_overbought']
                        if indicators['regime_params']
                        else 70  # Default
                    )
                    
                    entered = position_manager.enter_position(
                        entry_price=indicators['price'],
                        size=details['size'],
                        stop_loss=details['stop_loss'],
                        take_profit=details['take_profit'],
                        timestamp=indicators['timestamp'],
                        regime=indicators['regime'],
                        mfi_exit=mfi_exit
                    )
                    if entered:
                        print("\nPosition Entry:")
                        print(f"Entry Price: ${indicators['price']:.2f}")
                        print(f"Size: {details['size']} shares")
                        print(f"Stop Loss: ${details['stop_loss']:.2f}")
                        print(f"Take Profit: ${details['take_profit']:.2f}")
                        print(f"MFI Exit: {mfi_exit}")
            
            # Display current state
            format_output(indicators, position_manager)
            
            # Wait before next update
            print(f"Next update in {update_interval//60} minutes...")
            print(f"Current time (PST): {datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')}")
            sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")
        if position_manager.position_history:
            print("\nPosition History:")
            for pos in position_manager.position_history:
                print(f"\nEntry: ${pos['entry_price']:.2f}")
                print(f"Exit: ${pos['exit_price']:.2f}")
                print(f"PnL: ${pos['pnl']:.2f}")
                print(f"Reason: {pos['exit_reason']}")
                print(f"Regime: {pos['regime']}")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    monitor_live()
