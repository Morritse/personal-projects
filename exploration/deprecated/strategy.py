from typing import Tuple, Dict
import numpy as np
import time
import asyncio
from datetime import datetime
import pytz
from deprecated.data_fetcher import DataFetcher
from deprecated.indicators import IndicatorCalculator
from deprecated.trader import AlpacaTrader
from config import (
    TIMEFRAME_INDICATORS,
    UPDATE_INTERVAL,
    SYMBOLS,
    VERBOSE_DATA,
    POSITION_RISK_PERCENT,
    MAX_DAILY_LOSS,
    MAX_TRADES_PER_DAY,
    VOLUME_THRESHOLD,
    VOLATILITY_THRESHOLD,
    RELATIVE_VOLUME_THRESHOLD,
    SIGNAL_TYPES,
    STOP_TYPES
)
import pandas as pd

class TradingStrategy:
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.indicator_calculator = IndicatorCalculator()
        self.trader = AlpacaTrader()
        
        # Configuration
        self.allow_short_selling = False  # Disable short selling for this strategy
        self.market_open_hour = 9   # Market opens at 9:30 AM ET
        self.market_open_minute = 30
        self.market_close_hour = 16  # Market closes at 4:00 PM ET
        self.market_close_minute = 0
        
        # Track positions and trading stats
        self.positions = {symbol: {
            'side': 0,        # 0 = flat, 1 = long
            'size': 0,        # Number of shares
            'entry_price': 0, # Entry price for position
            'stop_price': 0,  # Current stop loss level
            'last_trade': None # Last trade timestamp
        } for symbol in SYMBOLS}
        
        # Daily tracking
        self.daily_stats = {
            'trades_today': 0,
            'daily_pnl': 0.0,
            'start_equity': 0.0
        }
        
        self.min_trade_interval = 60  # Minimum seconds between trades (1 minute)
        
        # Initialize positions from Alpaca
        for symbol in SYMBOLS:
            position = self.trader.get_position(symbol)
            if position > 0:
                self.positions[symbol]['side'] = 1
                self.positions[symbol]['size'] = position
            elif position < 0:
                self.positions[symbol]['side'] = -1
                self.positions[symbol]['size'] = abs(position)

    def _check_liquidity(self, symbol: str, data: pd.DataFrame) -> bool:
        """Check if symbol meets liquidity requirements."""
        avg_volume = data['volume'].mean()
        return avg_volume >= VOLUME_THRESHOLD

    def _check_volatility(self, data: pd.DataFrame) -> bool:
        """Check if price movement meets volatility threshold."""
        daily_range = (data['high'].max() - data['low'].min()) / data['close'].mean()
        return daily_range >= VOLATILITY_THRESHOLD

    def _check_volume_spike(self, current_volume: float, avg_volume: float) -> bool:
        """Check if current volume represents a spike."""
        return current_volume >= (avg_volume * RELATIVE_VOLUME_THRESHOLD)

    def _generate_decision(self, data: pd.DataFrame, signals: Dict[str, float], current_position: int) -> Tuple[str, float, float]:
        """Generate trading decision based on price action and indicators."""
        if not self._check_liquidity(symbol, data) or not self._check_volatility(data):
            return "HOLD", 0, 0

        current_price = data['close'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].mean()
        
        # Get indicator values
        vwap = signals.get('VWAP', 0)
        rsi = signals.get('RSI', 0)
        ema_9 = signals.get('EMA_9', 0)
        ema_20 = signals.get('EMA_20', 0)
        upper_donchian = signals.get('DONCHIAN_UPPER', 0)
        lower_donchian = signals.get('DONCHIAN_LOWER', 0)
        atr = signals.get('ATR', 0)

        # Check for breakout setup
        if (current_position == 0 and 
            current_price > upper_donchian and
            current_price > vwap and
            rsi > SIGNAL_TYPES['BREAKOUT']['rsi_minimum'] and
            self._check_volume_spike(current_volume, avg_volume)):
            return "BUY", current_price, atr

        # Check for pullback setup
        if (current_position == 0 and
            current_price > vwap and
            current_price > ema_20 and  # Uptrend
            abs(current_price - ema_9) / current_price < 0.001 and  # Close to 9 EMA
            self._check_volume_spike(current_volume, avg_volume)):
            return "BUY", current_price, atr

        # Exit signals
        if current_position > 0:
            # Stop loss hit
            if current_price < self.positions[symbol]['stop_price']:
                return "SELL", current_price, 0
            
            # Trail stop behind 9 EMA
            new_stop = ema_9 * (1 - STOP_TYPES['TRAILING']['buffer'])
            if new_stop > self.positions[symbol]['stop_price']:
                self.positions[symbol]['stop_price'] = new_stop

        return "HOLD", 0, 0

    def update_and_decide(self, symbol: str) -> Dict[str, any]:
        """Update market data and generate trading decision for a symbol."""
        # Check daily loss limit
        if self.daily_stats['daily_pnl'] <= -MAX_DAILY_LOSS * self.daily_stats['start_equity']:
            if VERBOSE_DATA:
                print(f"\nSkipping trades - Daily loss limit reached")
            return {
                "decision": "HOLD",
                "error": "Daily loss limit reached",
                "position": self._get_position_info(symbol)
            }
            
        # Check max trades per day
        if self.daily_stats['trades_today'] >= MAX_TRADES_PER_DAY:
            if VERBOSE_DATA:
                print(f"\nSkipping trades - Maximum daily trades reached")
            return {
                "decision": "HOLD",
                "error": "Maximum daily trades reached",
                "position": self._get_position_info(symbol)
            }

        # Fetch latest data for primary and trend timeframes
        timeframe_data = self.data_fetcher.update_data(symbol)
        if not timeframe_data:
            return {
                "decision": "HOLD",
                "error": f"Failed to fetch market data for {symbol}",
                "position": self._get_position_info(symbol)
            }

        # Calculate indicators for each timeframe
        primary_signals = self.indicator_calculator.get_indicator_signals(
            timeframe_data['primary'], 
            timeframe='primary'
        )
        trend_signals = self.indicator_calculator.get_indicator_signals(
            timeframe_data['trend'],
            timeframe='trend'
        )

        # Generate trading decision
        current_position = self.positions[symbol]['side']
        decision, entry_price, atr = self._generate_decision(
            timeframe_data['primary'],
            primary_signals,
            current_position
        )

        # Check if enough time has passed since last trade
        current_time = time.time()
        last_trade_time = self.positions[symbol]['last_trade']
        if last_trade_time and current_time - last_trade_time < self.min_trade_interval:
            if VERBOSE_DATA:
                print(f"\nSkipping trade for {symbol} - minimum interval not met")
            return {
                "decision": "HOLD",
                "position": self._get_position_info(symbol)
            }

        # Execute trades based on decision
        if decision == "BUY" and current_position == 0:
            # Calculate position size based on ATR
            account_value = float(self.trader.client.get_account().equity)
            risk_amount = account_value * POSITION_RISK_PERCENT
            stop_price = entry_price - (STOP_TYPES['FIXED'] * atr)
            shares = int(risk_amount / (entry_price - stop_price))
            
            if VERBOSE_DATA:
                print(f"\nPlacing BUY order for {symbol}")
                print(f"Entry: ${entry_price:.2f}, Stop: ${stop_price:.2f}, Shares: {shares}")
                
            if self.trader.submit_order(symbol, "BUY", qty=shares, limit_price=entry_price):
                self.positions[symbol].update({
                    'side': 1,
                    'size': shares,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'last_trade': time.time()
                })
                self.daily_stats['trades_today'] += 1
                
        elif decision == "SELL" and current_position > 0:
            if VERBOSE_DATA:
                print(f"\nClosing LONG position for {symbol}")
            self.trader.close_position(symbol)
            self.positions[symbol].update({
                'side': 0,
                'size': 0,
                'entry_price': 0,
                'stop_price': 0,
                'last_trade': time.time()
            })

        return {
            "decision": decision,
            "position": self._get_position_info(symbol)
        }

    def _get_position_info(self, symbol: str) -> Dict[str, any]:
        """Get formatted position information."""
        position = self.positions[symbol]
        return {
            "side": "LONG" if position['side'] > 0 else "FLAT",
            "size": position['size'],
            "entry_price": position['entry_price'],
            "stop_price": position['stop_price'],
            "last_trade": position['last_trade']
        }

    def _reset_daily_stats(self):
        """Reset daily trading statistics."""
        account = self.trader.client.get_account()
        self.daily_stats.update({
            'trades_today': 0,
            'daily_pnl': 0.0,
            'start_equity': float(account.equity)
        })

    async def run(self, callback=None):
        """Run the strategy continuously."""
        print(f"Starting trading strategy (Update interval: {UPDATE_INTERVAL}s)")
        print(f"Trading hours: {self.market_open_hour}:{self.market_open_minute:02d} - "
              f"{self.market_close_hour}:{self.market_close_minute:02d} ET")
        print(f"Monitoring symbols: {', '.join(SYMBOLS)}")
        print("\nRisk Management:")
        print(f"Position Risk: {POSITION_RISK_PERCENT*100}% per trade")
        print(f"Daily Loss Limit: {MAX_DAILY_LOSS*100}%")
        print(f"Max Daily Trades: {MAX_TRADES_PER_DAY}")

        # Initialize daily stats
        self._reset_daily_stats()
        last_trading_day = datetime.now().date()

        while True:
            try:
                # Show market status
                market_open = self._is_market_open()
                status = "OPEN" if market_open else "CLOSED"
                print(f"\n=== Market Analysis ({status}) ===")
                
                # Fetch all data concurrently
                all_data = await self.data_fetcher.update_data()
                
                # Check for new trading day
                current_date = datetime.now().date()
                if current_date != last_trading_day:
                    self._reset_daily_stats()
                    last_trading_day = current_date

                # Process all symbols
                results = {}
                for symbol, timeframe_data in all_data.items():
                    try:
                        result = self.update_and_decide(symbol)
                        results[symbol] = result
                        
                        # Print status
                        position = result['position']
                        print(f"{symbol:<5} | {result['decision']:<4} | "
                              f"Pos: {position['side']:<5} | "
                              f"Size: {position['size']} | "
                              f"Entry: ${position['entry_price']:.2f} | "
                              f"Stop: ${position['stop_price']:.2f}")
                              
                    except Exception as e:
                        print(f"{symbol:<5} | Error: {str(e)}")
                        continue
                
                if not results:
                    raise Exception("No valid results from any symbols")
                
                # Print active positions
                active_positions = [
                    f"{symbol}: {result['position']['side']} position, "
                    f"Size: {result['position']['size']}, "
                    f"Entry: ${result['position']['entry_price']:.2f}, "
                    f"Stop: ${result['position']['stop_price']:.2f}"
                    for symbol, result in results.items()
                    if result['position']['size'] > 0
                ]
                
                if active_positions:
                    print("\nActive Positions:")
                    for position in active_positions:
                        print(position)
                
                print(f"\nNext update in {UPDATE_INTERVAL}s...")
                await asyncio.sleep(UPDATE_INTERVAL)
                
            except KeyboardInterrupt:
                print("\nStopping trading strategy...")
                break
            except Exception as e:
                print(f"\nError in strategy execution: {str(e)}")
                print("Waiting before retry...")
                await asyncio.sleep(UPDATE_INTERVAL)

    def _is_market_open(self) -> bool:
        """Check if the market is currently open using Alpaca API."""
        try:
            # Get real-time market status from Alpaca
            clock = self.trader.client.get_clock()
            if not clock.is_open:
                if VERBOSE_DATA:
                    print("Market closed - Alpaca reports market is closed")
                return False
                
            return True
            
        except Exception as e:
            if VERBOSE_DATA:
                print(f"Error checking market status: {str(e)}")
                print("Falling back to time-based check")
                
            # Fallback to time-based check if API fails
            now = datetime.now()
            
            # Check if it's a weekday (0 = Monday, 6 = Sunday)
            if now.weekday() > 4:  # Saturday or Sunday
                if VERBOSE_DATA:
                    print("Market closed - Weekend")
                return False
                
            # Add 3 hours to convert PST to EST
            est_hour = (now.hour + 3) % 24
            est_date = now.date()
            
            # Holiday early closures (1:00 PM EST)
            holiday_early_closes = [
                # 2024 holidays
                datetime(2024, 7, 3).date(),    # July 3rd (Independence Day)
                datetime(2024, 11, 29).date(),  # Black Friday
                datetime(2024, 12, 24).date(),  # Christmas Eve
            ]
            
            # Full holiday closures
            market_holidays = [
                # 2024 holidays
                datetime(2024, 1, 1).date(),    # New Year's Day
                datetime(2024, 1, 15).date(),   # Martin Luther King Jr. Day
                datetime(2024, 2, 19).date(),   # Presidents Day
                datetime(2024, 3, 29).date(),   # Good Friday
                datetime(2024, 5, 27).date(),   # Memorial Day
                datetime(2024, 6, 19).date(),   # Juneteenth
                datetime(2024, 7, 4).date(),    # Independence Day
                datetime(2024, 9, 2).date(),    # Labor Day
                datetime(2024, 11, 28).date(),  # Thanksgiving Day
                datetime(2024, 12, 25).date(),  # Christmas Day
            ]
            
            # Check for full holiday closure
            if est_date in market_holidays:
                if VERBOSE_DATA:
                    print("Market closed - Holiday")
                return False
                
            # Convert current EST time to minutes since midnight
            current_minutes = est_hour * 60 + now.minute
            market_open_minutes = self.market_open_hour * 60 + self.market_open_minute
            
            # Adjust close time for early closure days (1:00 PM EST)
            if est_date in holiday_early_closes:
                market_close_minutes = 13 * 60  # 1:00 PM EST
                if VERBOSE_DATA:
                    print("Early market closure today (1:00 PM ET)")
            else:
                market_close_minutes = self.market_close_hour * 60 + self.market_close_minute
            
            # Check if within market hours
            is_open = market_open_minutes <= current_minutes < market_close_minutes
        
        if not is_open and VERBOSE_DATA:
            print(f"Market closed - Outside trading hours ({self.market_open_hour}:{self.market_open_minute:02d} - {self.market_close_hour}:{self.market_close_minute:02d} ET)")
            
        return is_open
