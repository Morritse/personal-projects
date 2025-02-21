import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv
from strategy.vwap_obv_strategy_optimizer import VWAPOBVCrossover, strategy_configuration

class LiveTrader:
    def __init__(self):
        """Initialize live trader with Alpaca API."""
        load_dotenv()
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        base_url = 'https://paper-api.alpaca.markets'  # Paper trading
        
        if not api_key or not api_secret:
            raise ValueError("API credentials not found in environment variables")
        
        self.api = tradeapi.REST(
            api_key,
            api_secret,
            base_url=base_url,
            api_version='v2'
        )
        
        # Initialize strategy
        self.strategy = VWAPOBVCrossover(strategy_configuration)
        self.position = None
        self.last_check = None
        
        # Trading parameters
        self.symbols = ['JNJ', 'XLV']  # Stock and sector ETF
        self.timeframe = '1Hour'
        self.lookback_hours = 50  # For indicators
        
        print("\nLive trader initialized")
        print(f"Strategy config: {strategy_configuration}")
    
    def get_historical_data(self, symbol: str, hours: int = 50) -> pd.DataFrame:
        """Get historical hourly data for calculating indicators."""
        end = datetime.now()
        start = end - timedelta(hours=hours)
        
        bars = self.api.get_bars(
            symbol,
            tradeapi.TimeFrame.Hour,
            start.strftime('%Y-%m-%d'),
            end.strftime('%Y-%m-%d'),
            adjustment='raw'
        ).df
        
        return bars
    
    def check_market_hours(self) -> bool:
        """Check if market is open and within trading hours."""
        clock = self.api.get_clock()
        if not clock.is_open:
            return False
        
        current_time = datetime.now().time()
        return (
            self.strategy.morning_start <= current_time < self.strategy.market_close
        )
    
    def get_position(self) -> Optional[Dict]:
        """Get current position details."""
        try:
            position = self.api.get_position('JNJ')
            return {
                'qty': int(position.qty),
                'entry_price': float(position.avg_entry_price)
            }
        except:
            return None
    
    def analyze_regime(self, xlv_data: pd.DataFrame) -> Dict:
        """Analyze current market regime."""
        returns = xlv_data['close'].pct_change()
        window = self.strategy.regime_window
        
        # Calculate metrics
        ret = returns.rolling(window=window).mean() * 252
        vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        current_ret = ret.iloc[-1]
        current_vol = vol.iloc[-1]
        
        # Get volatility thresholds
        vol_33pct = vol.quantile(0.33)
        vol_67pct = vol.quantile(0.67)
        
        # Determine regime
        if current_ret > 0:
            if current_vol > vol_67pct:
                regime = 'bull_high_vol'
            elif current_vol < vol_33pct:
                regime = 'bull_low_vol'
            else:
                regime = 'bull_med_vol'
        else:
            if current_vol > vol_67pct:
                regime = 'bear_high_vol'
            elif current_vol < vol_33pct:
                regime = 'bear_low_vol'
            else:
                regime = 'bear_med_vol'
        
        return {
            'regime': regime,
            'return': current_ret * 100,
            'volatility': current_vol * 100,
            'vol_33pct': vol_33pct * 100,
            'vol_67pct': vol_67pct * 100
        }
    
    def execute_entry(self, signal: Dict) -> bool:
        """Execute entry order."""
        try:
            # Calculate order size
            account = self.api.get_account()
            equity = float(account.equity)
            size = signal['size']
            
            # Submit order
            order = self.api.submit_order(
                symbol='JNJ',
                qty=size,
                side='buy',
                type='limit',
                time_in_force='day',
                limit_price=round(signal['price'] * 1.001, 2),  # 0.1% slippage buffer
                stop_loss={
                    'stop_price': round(signal['stop_loss'], 2),
                    'limit_price': round(signal['stop_loss'] * 0.999, 2)  # 0.1% slippage
                },
                take_profit={
                    'limit_price': round(signal['take_profit'], 2)
                }
            )
            
            print(f"\nEntry order submitted:")
            print(f"Size: {size} shares")
            print(f"Price: ${signal['price']:.2f}")
            print(f"Stop: ${signal['stop_loss']:.2f}")
            print(f"Target: ${signal['take_profit']:.2f}")
            print(f"Regime: {signal['regime']}")
            
            return True
            
        except Exception as e:
            print(f"Entry order failed: {e}")
            return False
    
    def execute_exit(self, signal: Dict) -> bool:
        """Execute exit order."""
        try:
            position = self.get_position()
            if not position:
                return False
            
            # Submit order
            order = self.api.submit_order(
                symbol='JNJ',
                qty=position['qty'],
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            print(f"\nExit order submitted:")
            print(f"Size: {position['qty']} shares")
            print(f"Reason: {signal['reason']}")
            
            return True
            
        except Exception as e:
            print(f"Exit order failed: {e}")
            return False
    
    def check_signals(self):
        """Check for entry/exit signals."""
        # Get latest data
        jnj_data = self.get_historical_data('JNJ', self.lookback_hours)
        xlv_data = self.get_historical_data('XLV', self.lookback_hours)
        
        # Analyze current regime
        regime_analysis = self.analyze_regime(xlv_data)
        print("\nCurrent Market Regime:")
        print(f"Regime: {regime_analysis['regime']}")
        print(f"Return (ann.): {regime_analysis['return']:.1f}%")
        print(f"Volatility (ann.): {regime_analysis['volatility']:.1f}%")
        print(f"Vol Thresholds: {regime_analysis['vol_33pct']:.1f}% - {regime_analysis['vol_67pct']:.1f}%")
        
        position = self.get_position()
        
        if position:
            # Check exit
            exit_signal, exit_data = self.strategy.check_exit(jnj_data, xlv_data, {
                'entry_time': pd.Timestamp(datetime.now() - timedelta(hours=1)),  # Approximate
                'entry_price': position['entry_price']
            })
            
            if exit_signal:
                success = self.execute_exit(exit_data)
                if success:
                    self.position = None
        
        else:
            # Check entry
            entry_signal, entry_data = self.strategy.check_entry(jnj_data, xlv_data)
            
            if entry_signal:
                success = self.execute_entry(entry_data)
                if success:
                    self.position = entry_data
    
    def run(self):
        """Main trading loop."""
        print("\nStarting live trading...")
        print("Checking signals hourly during market hours")
        print("Press Ctrl+C to stop")
        
        while True:
            try:
                now = datetime.now()
                
                # Check once per hour during market hours
                if (self.last_check is None or 
                    (now - self.last_check).total_seconds() >= 3600):
                    
                    if self.check_market_hours():
                        print(f"\nChecking signals at {now}")
                        self.check_signals()
                        self.last_check = now
                    
                    else:
                        print(f"\nMarket closed at {now}")
                        self.last_check = now
                
                # Sleep for 1 minute before next check
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\nStopping live trader...")
                break
            except Exception as e:
                print(f"\nError in main loop: {e}")
                continue

if __name__ == "__main__":
    trader = LiveTrader()
    trader.run()
