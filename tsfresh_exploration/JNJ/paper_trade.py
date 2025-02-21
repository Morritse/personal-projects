import os
import json
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from strategy.vwap_obv_strategy import VWAPOBVCrossover
from time import sleep
import pytz

# Load environment variables
load_dotenv()

class PaperTrader:
    def __init__(self):
        # Initialize API clients
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        if not api_key or not secret_key:
            raise ValueError("Alpaca API credentials not found in .env file")
        
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        
        # Load strategy config
        with open('final_strategy_params.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize strategy
        self.strategy = VWAPOBVCrossover(self.config)
        self.position = None
        self.trades = []
    
    def get_position(self, symbol):
        """Get current position details."""
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                if pos.symbol == symbol:
                    return pos
            return None
        except Exception as e:
            print(f"Error getting position: {e}")
            return None
    
    def submit_order(self, symbol, qty, side, take_profit=None, stop_loss=None):
        """Submit a new order with optional take profit and stop loss."""
        try:
            # Create bracket order
            order_data = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": OrderType.MARKET,
                "time_in_force": TimeInForce.DAY
            }
            
            # Add bracket legs if entry order
            if side == OrderSide.BUY and take_profit and stop_loss:
                order_data["take_profit"] = {"limit_price": take_profit}
                order_data["stop_loss"] = {"stop_price": stop_loss}
            
            # Submit order
            order = self.trading_client.submit_order(
                MarketOrderRequest(**order_data)
            )
            
            print(f"\nOrder submitted: {side} {qty} {symbol}")
            if take_profit and stop_loss:
                print(f"Take profit: ${take_profit:.2f}")
                print(f"Stop loss: ${stop_loss:.2f}")
            
            return order
            
        except Exception as e:
            print(f"Error submitting order: {e}")
            return None
    
    def fetch_data(self, symbol, lookback_days=30):
        """Fetch recent market data."""
        pst = pytz.timezone('US/Pacific')
        utc = pytz.UTC
        
        end = datetime.now(pst).astimezone(utc)
        start = end - timedelta(days=lookback_days)
        
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Hour,
            start=start,
            end=end,
            adjustment='raw'
        )
        
        try:
            bars = self.data_client.get_stock_bars(request)
            df = bars.df
            
            if len(df) == 0:
                raise ValueError(f"No data returned for {symbol}")
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol]
            
            df.index = df.index.tz_convert(pst)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise
    
    def check_entry(self, jnj_data, xlv_data):
        """Check for entry conditions and execute trade if valid."""
        indicators = self.calculate_indicators(jnj_data, xlv_data)
        
        if all(indicators['conditions'].values()):
            details = indicators['position_details']
            if details:
                # Submit entry order with brackets
                order = self.submit_order(
                    symbol='JNJ',
                    qty=details['size'],
                    side=OrderSide.BUY,
                    take_profit=details['take_profit'],
                    stop_loss=details['stop_loss']
                )
                
                if order:
                    self.position = {
                        'entry_price': indicators['price'],
                        'size': details['size'],
                        'stop_loss': details['stop_loss'],
                        'take_profit': details['take_profit'],
                        'entry_time': indicators['timestamp'],
                        'regime': indicators['regime'],
                        'highest_price': indicators['price'],
                        'mfi_exit': indicators['regime_params']['mfi_overbought']
                    }
                    return True
        
        return False
    
    def check_exit(self, current_price, timestamp, vwap, mfi):
        """Check for exit conditions and execute if needed."""
        position = self.get_position('JNJ')
        if not position:
            return None
        
        # Check technical exits (VWAP cross or MFI)
        if current_price > vwap or mfi > self.position['mfi_exit']:
            # Submit market exit order
            order = self.submit_order(
                symbol='JNJ',
                qty=position.qty,
                side=OrderSide.SELL
            )
            
            if order:
                pnl = (current_price - float(position.avg_entry_price)) * float(position.qty)
                exit_data = {
                    'exit_price': current_price,
                    'exit_time': timestamp,
                    'pnl': pnl,
                    'exit_reason': 'technical'
                }
                self.trades.append(exit_data)
                self.position = None
                return exit_data
        
        return None
    
    def calculate_indicators(self, jnj_data, xlv_data):
        """Calculate strategy indicators."""
        current_jnj = jnj_data.iloc[-1]
        price = current_jnj['close']
        
        vwap = self.strategy.calculate_vwap(jnj_data, self.strategy.vwap_window).iloc[-1]
        mfi = self.strategy.calculate_mfi(jnj_data).iloc[-1]
        obv = self.strategy.calculate_obv(jnj_data)
        obv_change = obv.diff().iloc[-1]
        regime = self.strategy.classify_regime(xlv_data)
        
        price_below_vwap = price < vwap
        obv_falling = obv_change < 0
        mfi_oversold = mfi < self.strategy.mfi_entry
        
        regime_params = self.strategy.get_regime_parameters(regime) if regime else None
        
        position_size = None
        stop_loss = None
        take_profit = None
        
        if all([price_below_vwap, obv_falling, mfi_oversold, regime]):
            params = regime_params
            atr = self.strategy.calculate_atr(jnj_data).iloc[-1]
            raw_stop = params['stop_mult'] * atr
            stop_distance = min(max(raw_stop, self.strategy.min_stop_dollars), self.strategy.max_stop_dollars)
            
            account = self.trading_client.get_account()
            capital = float(account.buying_power)
            risk_amount = capital * self.strategy.risk_per_trade
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
    
    def format_output(self, indicators):
        """Format current market state."""
        print("\n" + "="*50)
        print(f"Time (PST): {indicators['timestamp']}")
        print("-"*50)
        
        # Show active position if any
        position = self.get_position('JNJ')
        if position:
            unrealized_pnl = float(position.unrealized_pl)
            print("\nActive Position:")
            print(f"Size: {position.qty} shares")
            print(f"Entry Price: ${float(position.avg_entry_price):.2f}")
            print(f"Current Price: ${indicators['price']:.2f}")
            print(f"Unrealized PnL: ${unrealized_pnl:.2f}")
            if self.position:
                print(f"Stop Loss: ${self.position['stop_loss']:.2f}")
                print(f"Take Profit: ${self.position['take_profit']:.2f}")
                print(f"Regime: {self.position['regime']}")
                print(f"MFI Exit: {self.position['mfi_exit']}")
        
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
        
        # Only show entry signal if no position
        if not position:
            entry_signal = all(conditions.values())
            print("\nSignal:", "🟢 ENTRY" if entry_signal else "⚪ WAIT")
            
            if entry_signal and indicators['position_details']:
                details = indicators['position_details']
                print("\nPotential Position:")
                print(f"Size: {details['size']} shares")
                print(f"Stop Loss: ${details['stop_loss']:.2f}")
                print(f"Take Profit: ${details['take_profit']:.2f}")
        
        print("="*50 + "\n")
    
    def run(self):
        """Run paper trading strategy."""
        update_interval = 300  # 5 minutes
        pst = pytz.timezone('US/Pacific')
        
        print("Starting paper trading...")
        print("Press Ctrl+C to exit")
        print(f"Updating every {update_interval//60} minutes")
        print(f"Current time (PST): {datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        try:
            while True:
                now = datetime.now(pst).time()
                
                # Only trade during market hours (PST)
                if now < time(6, 30) or now > time(13, 0):
                    print("\nMarket closed. Waiting for next session...")
                    next_open = datetime.now(pst)
                    if now > time(13, 0):
                        next_open += timedelta(days=1)
                    next_open = next_open.replace(hour=6, minute=30, second=0)
                    sleep_seconds = (next_open - datetime.now(pst)).total_seconds()
                    sleep(sleep_seconds)
                    continue
                
                # Fetch latest data
                print("\nFetching latest data...")
                jnj_data = self.fetch_data('JNJ')
                xlv_data = self.fetch_data('XLV')
                
                # Calculate indicators
                indicators = self.calculate_indicators(jnj_data, xlv_data)
                
                # Check for position exit
                if self.get_position('JNJ'):
                    exit_data = self.check_exit(
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
                elif self.check_entry(jnj_data, xlv_data):
                    print("\nEntered new position")
                
                # Display current state
                self.format_output(indicators)
                
                # Wait before next update
                print(f"Next update in {update_interval//60} minutes...")
                print(f"Current time (PST): {datetime.now(pst).strftime('%Y-%m-%d %H:%M:%S %Z')}")
                sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\nPaper trading stopped by user")
            if self.trades:
                print("\nTrade History:")
                for trade in self.trades:
                    print(f"\nEntry: ${trade['entry_price']:.2f}")
                    print(f"Exit: ${trade['exit_price']:.2f}")
                    print(f"PnL: ${trade['pnl']:.2f}")
                    print(f"Reason: {trade['exit_reason']}")
                    print(f"Regime: {trade['regime']}")
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    trader = PaperTrader()
    trader.run()
