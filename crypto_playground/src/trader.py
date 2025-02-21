import asyncio
import json
import logging
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import os
from dotenv import load_dotenv
import websockets
from logging_config import setup_logging

from ensemble import EnsembleStrategy, SignalResult

# Setup logging
logger = setup_logging()

class MarketData:
    def __init__(self, data: Dict):
        # Extract message type and symbol
        self.type = data.get('T', data.get('type'))
        # Handle symbol in various formats
        if 'sym' in data:
            self.symbol = data['sym']
        elif 'S' in data:
            self.symbol = data['S']
        elif 'symbol' in data:
            self.symbol = data['symbol']
        else:
            # Try to extract from stream name
            stream = data.get('stream', '')
            if '.' in stream:
                self.symbol = stream.split('.')[-1]
            elif '_' in stream:
                self.symbol = stream.split('_')[-1]
            else:
                self.symbol = ''
        
        # Handle timestamp in various formats
        ts = data.get('t', data.get('timestamp'))
        if isinstance(ts, (int, float)):  # Unix timestamp
            self.timestamp = datetime.fromtimestamp(ts/1000, timezone.utc)
        else:  # ISO format
            self.timestamp = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        
        if self.type == 'b':  # Bar data
            self.open = float(data.get('o', 0))
            self.high = float(data.get('h', 0))
            self.low = float(data.get('l', 0))
            self.close = float(data.get('c', 0))
            self.volume = float(data.get('v', 0))
            self.trades = int(data.get('n', 0))
            self.vwap = float(data.get('vw', 0))
        elif self.type == 't':  # Trade data
            self.price = float(data.get('p', 0))
            self.size = float(data.get('s', 0))
            self.taker_side = data.get('tks', '')  # B for buyer, S for seller
            self.trade_id = data.get('i', '')
        elif self.type == 'q':  # Quote data
            self.bid_price = float(data.get('bp', 0))
            self.bid_size = float(data.get('bs', 0))
            self.ask_price = float(data.get('ap', 0))
            self.ask_size = float(data.get('as', 0))
        elif self.type == 'o':  # Orderbook data
            self.bids = [(float(b['p']), float(b['s'])) for b in data.get('b', [])]
            self.asks = [(float(a['p']), float(a['s'])) for a in data.get('a', [])]
            self.reset = data.get('r', False)  # True if full orderbook reset

class CryptoTrader:
    def __init__(
        self,
        symbols: List[str] = ["BTC/USD", "ETH/USD"],
        bar_timeframe: int = 5,  # minutes
        lookback_periods: int = 100,
        update_interval: int = 1,  # minute for monitoring
        trade_interval: int = 5    # minutes between trade decisions
    ):
        # Load configuration
        load_dotenv()
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        
        # Initialize parameters
        self.symbols = symbols
        self.bar_timeframe = bar_timeframe
        self.lookback_periods = 200  # Need more data for proper indicator calculation
        self.update_interval = update_interval  # for signal monitoring
        self.trade_interval = trade_interval    # for trade decisions
        self.last_trade = datetime.now(timezone.utc)
        
        # Initialize clients
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True)
        self.data_client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
        
        # Initialize strategy
        self.strategy = EnsembleStrategy()
        
        # Initialize data storage
        self.bars_data: Dict[str, pd.DataFrame] = {}
        self.current_positions: Dict[str, Dict] = {}
        self.trade_volume: Dict[str, float] = {s: 0.0 for s in symbols}  # Track volume
        self.last_bar_time: Dict[str, datetime] = {}  # Track last bar time
        self.pending_orders: Dict[str, Dict] = {}  # Track pending orders
        
        # Initialize last update time
        self.last_analysis = datetime.now(timezone.utc)
        
        # Stream control
        self._running = False
        self._ws = None
        
    async def initialize(self):
        """Initialize the trader with historical data."""
        logger.info("Fetching historical data...")
        # Get historical data for each symbol
        for symbol in self.symbols:
            await self._fetch_historical_data(symbol)
            
        # Get current positions
        await self._update_positions()
        
        logger.info(f"Initialized trader for symbols: {', '.join(self.symbols)}")
        
    async def _fetch_historical_data(self, symbol: str):
        """Fetch historical bar data for initialization."""
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(self.bar_timeframe, TimeFrame.Minute),
            start=datetime.now(timezone.utc) - timedelta(days=60),  # Get more history (60 days)
            end=datetime.now(timezone.utc)
        )
        
        logger.info(f"Fetching historical data for {symbol} from {request.start} to {request.end}")
        
        bars = self.data_client.get_crypto_bars(request)
        df = pd.DataFrame([{
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        } for bar in bars[symbol]])
        
        df.set_index('timestamp', inplace=True)
        df = df.sort_index()  # Ensure chronological order
        
        # Store full history
        self.bars_data[symbol] = df
        logger.info(f"Fetched {len(df)} bars for {symbol}")
        
    def _update_bars_data(self, symbol: str, new_bar: pd.DataFrame):
        """Update historical bars data with new bar."""
        if symbol not in self.bars_data:
            self.bars_data[symbol] = new_bar
        else:
            # Add new bar and maintain chronological order
            self.bars_data[symbol] = pd.concat([
                self.bars_data[symbol],
                new_bar
            ]).sort_index()
            
            # Remove duplicate bars if any
            self.bars_data[symbol] = self.bars_data[symbol].loc[~self.bars_data[symbol].index.duplicated(keep='last')]
        
    def _get_analysis_data(self, symbol: str) -> Dict[str, np.ndarray]:
        """Get data arrays for analysis, using full history."""
        if symbol not in self.bars_data:
            raise ValueError(f"No data available for {symbol}")
            
        df = self.bars_data[symbol]
        
        # Skip first 20 bars for indicator warmup
        if len(df) <= 20:
            raise ValueError(f"Not enough data points for {symbol}")
            
        # Use last 30 days of data (like in backtesting)
        df = df.iloc[-self.lookback_periods:]
        
        # Check if current time is within active trading hours (8 AM to 8 PM UTC)
        current_hour = datetime.now(timezone.utc).hour
        if not (8 <= current_hour < 20):
            logger.info("Outside active trading hours (8 AM - 8 PM UTC)")
            return {
                'open': np.array([]),
                'high': np.array([]),
                'low': np.array([]),
                'close': np.array([]),
                'volume': np.array([])
            }
        
        # Calculate indicators
        close_series = pd.Series(df['close'].values)
        volume_series = pd.Series(df['volume'].values)
        
        # EMAs for trend detection
        ema10 = close_series.ewm(span=10, adjust=False).mean()
        ema20 = close_series.ewm(span=20, adjust=False).mean()
        ema50 = close_series.ewm(span=50, adjust=False).mean()
        ema100 = close_series.ewm(span=100, adjust=False).mean()
        
        # Trend alignment
        st_trend = pd.Series((ema10.values / ema20.values - 1) * 100)
        lt_trend = pd.Series((ema50.values / ema100.values - 1) * 100)
        trend_aligned = np.array([np.sign(st_trend.iloc[-1]) == np.sign(lt_trend.iloc[-1])])
        
        # Volume trend
        volume_ma = volume_series.rolling(window=20).mean().values
        volume_trend = np.where(volume_ma > 0, volume_series.values / volume_ma, 0)  # Handle zero division
        volume_filter = np.where((volume_trend > 1.1) & (volume_trend < 2.5), True, False)
        
        # Momentum score
        momentum = close_series.pct_change(5)
        mom_std = momentum.rolling(window=20).std()
        momentum_score = (momentum / mom_std).clip(-2, 2)
        
        # Volatility for stop loss/take profit
        volatility = close_series.pct_change().rolling(window=20).std()
        vol_ratio = volatility / volatility.rolling(window=100).mean()
        
        # Dynamic ATR multipliers
        stop_mult = np.where(vol_ratio > 1.5, 1.0,  # Tighter stops in high vol
                           np.where(vol_ratio < 0.5, 1.5,  # Wider stops in low vol
                                  1.2))[-1]  # Normal regime
        tp_mult = stop_mult * 1.5  # Maintain good risk/reward ratio
        
        return {
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'close': df['close'].values,
            'volume': df['volume'].values,
            'ema10': ema10.values,
            'ema20': ema20.values,
            'ema50': ema50.values,
            'ema100': ema100.values,
            'trend_aligned': trend_aligned,
            'volume_filter': volume_filter,
            'momentum_score': momentum_score.values,
            'stop_mult': stop_mult,
            'tp_mult': tp_mult
        }
        
    async def _handle_market_data(self, msg: Dict):
        """Process incoming market data."""
        try:
            data = MarketData(msg)
            
            if data.type == 'b':  # Bar data
                # Enhanced bar data logging
                logger.info(f"\n{'='*50}")
                logger.info(f"Bar Update - {data.symbol} at {data.timestamp}")
                logger.info(f"{'='*50}")
                logger.info(f"Price Action:")
                logger.info(f"  Open: ${data.open:,.2f}")
                logger.info(f"  High: ${data.high:,.2f}")
                logger.info(f"  Low:  ${data.low:,.2f}")
                logger.info(f"  Close: ${data.close:,.2f}")
                logger.info(f"\nVolume Analysis:")
                logger.info(f"  Volume: {data.volume:,.4f}")
                if data.trades > 0:
                    logger.info(f"  Trades: {data.trades}")
                    if data.volume > 0:
                        logger.info(f"  VWAP: ${data.vwap:,.2f}")
                
                # Calculate and log market conditions
                try:
                    # Get historical data for volatility calculation
                    if data.symbol in self.bars_data:
                        hist_data = self.bars_data[data.symbol]['close'].values
                        returns = np.diff(np.log(hist_data))
                        if len(returns) > 0:
                            current_vol = np.std(returns[-20:] if len(returns) >= 20 else returns)
                            historical_vol = np.std(returns[-100:] if len(returns) >= 100 else returns)
                            vol_ratio = current_vol / historical_vol if historical_vol != 0 else 1.0
                        else:
                            vol_ratio = 1.0
                    else:
                        vol_ratio = 1.0
                except Exception as e:
                    logger.error(f"Error calculating volatility: {str(e)}")
                    vol_ratio = 1.0
                
                logger.info(f"\nMarket Conditions:")
                logger.info(f"  Volatility Ratio: {vol_ratio:.2f}")
                if vol_ratio > 1.5:
                    logger.info("  Regime: HIGH VOLATILITY")
                elif vol_ratio < 0.5:
                    logger.info("  Regime: LOW VOLATILITY")
                else:
                    logger.info("  Regime: NORMAL VOLATILITY")
                
                # Update bars data
                new_bar = pd.DataFrame([{
                    'timestamp': data.timestamp,
                    'open': data.open,
                    'high': data.high,
                    'low': data.low,
                    'close': data.close,
                    'volume': data.volume + self.trade_volume[data.symbol]  # Add accumulated volume
                }]).set_index('timestamp')
                
                # Reset accumulated volume
                self.trade_volume[data.symbol] = 0.0
                
                # Update historical data
                self._update_bars_data(data.symbol, new_bar)
                self.last_bar_time[data.symbol] = data.timestamp
                
                # Check if enough time has passed since last trade
                now = datetime.now(timezone.utc)
                time_since_last_trade = (now - self.last_trade).total_seconds() / 60  # in minutes
                
                # Always analyze, but only trade if enough time has passed
                monitor_only = time_since_last_trade < self.trade_interval
                await self._analyze_and_trade(data.symbol, monitor_only=monitor_only)
                
                # Update timestamps
                self.last_analysis = now
                if not monitor_only:
                    self.last_trade = now
                
            elif data.type == 't':  # Trade data
                # Accumulate volume
                self.trade_volume[data.symbol] += data.size
                
                # Log significant trades
                if data.price * data.size > 100000:  # $100k+ trades
                    logger.info(f"\nLarge trade in {data.symbol}:")
                    logger.info(f"Price: ${data.price:,.2f}")
                    logger.info(f"Size: {data.size:.4f}")
                    logger.info(f"Value: ${data.price * data.size:,.2f}")
                    if data.taker_side:
                        logger.info(f"Taker: {'Buyer' if data.taker_side == 'B' else 'Seller'}")
                
                # Check stop loss and take profit if we have a position
                if data.symbol in self.current_positions:
                    position = self.current_positions[data.symbol]
                    # Check stop loss and take profit based on position side
                    if position['side'] == 'long':
                        # Long position: stop loss below entry, take profit above
                        if data.price <= position['stop_loss']:
                            await self._close_position(data.symbol, 'Stop loss hit')
                        elif data.price >= position['take_profit']:
                            await self._close_position(data.symbol, 'Take profit hit')
                    else:  # short position
                        # Short position: stop loss above entry, take profit below
                        if data.price >= position['stop_loss']:
                            await self._close_position(data.symbol, 'Stop loss hit')
                        elif data.price <= position['take_profit']:
                            await self._close_position(data.symbol, 'Take profit hit')
                            
            elif data.type == 'u':  # Updated bar data
                logger.info(f"\nUpdated bar for {data.symbol}:")
                logger.info(f"Time: {data.timestamp}")
                logger.info(f"OHLCV: {data.open:.2f}, {data.high:.2f}, {data.low:.2f}, {data.close:.2f}, {data.volume:.4f}")
                if data.trades > 0:
                    logger.info(f"Additional trades: {data.trades}")
                    if data.volume > 0:
                        logger.info(f"Updated VWAP: {data.vwap:.2f}")
                
                # Update historical data
                new_bar = pd.DataFrame([{
                    'timestamp': data.timestamp,
                    'open': data.open,
                    'high': data.high,
                    'low': data.low,
                    'close': data.close,
                    'volume': data.volume
                }]).set_index('timestamp')
                
                self._update_bars_data(data.symbol, new_bar)
                
        except Exception as e:
            logger.error(f"Error processing market data: {str(e)}")
                    
    async def _analyze_and_trade(self, symbol: str, monitor_only: bool = False):
        """Analyze current data and execute trades if conditions are met."""
        try:
            # Get analysis data using full history
            data = self._get_analysis_data(symbol)
            current_price = data['close'][-1]
            
            # Get trading signals
            signal = self.strategy.analyze(data, current_price)
            trade_params = None
            
            # Only proceed with analysis if we have valid signals
            if signal and signal.trend_signal and signal.momentum_signal and signal.volatility_signal:
                # Log signal analysis
                self._log_signal_analysis(symbol, current_price, signal)
                
                # Get trade parameters
                trade_params = self.strategy.get_trade_params()
                
                if trade_params and not monitor_only:
                    logger.info(f"\n{'='*50}")
                    logger.info(f"Trade Signal - {symbol}")
                    logger.info(f"{'='*50}")
                    logger.info(f"Signal Strength: {trade_params['signal']:.3f}")
                    logger.info(f"Confidence: {trade_params['confidence']:.3f}")
                    logger.info(f"Position Size: {trade_params['position_size']*100:.1f}% of capital")
                    logger.info(f"Risk Management:")
                    logger.info(f"  Stop Loss: ${trade_params['stop_loss']:,.2f}")
                    logger.info(f"  Take Profit: ${trade_params['take_profit']:,.2f}")
                    logger.info(f"  Risk/Reward: {(trade_params['take_profit']-current_price)/(current_price-trade_params['stop_loss']):.2f}")
                    
                    # Check for entry signals
                    if symbol not in self.current_positions:
                        logger.info(f"\nEvaluating entry conditions for {symbol}:")
                        logger.info(f"Current positions: {list(self.current_positions.keys())}")
                        logger.info(f"Signal: {trade_params['signal']}")
                        
                        # Long signal conditions
                        if trade_params['signal'] > self.strategy.MIN_SIGNAL_STRENGTH:
                            logger.info("Long signal detected - attempting entry")
                            await self._enter_position(symbol, 'buy', trade_params)
                        # Short signal conditions
                        elif trade_params['signal'] < -self.strategy.MIN_SIGNAL_STRENGTH:
                            logger.info("Short signal detected - attempting entry")
                            await self._enter_position(symbol, 'sell', trade_params)
                        else:
                            logger.info("No valid entry signal")
                    # Check for exit signals
                    else:
                        position = self.current_positions[symbol]
                        # Exit long if signal turns negative
                        if position['side'] == 'long' and trade_params['signal'] < 0:
                            await self._close_position(symbol, 'Signal turned negative')
                        # Exit short if signal turns positive
                        elif position['side'] == 'short' and trade_params['signal'] > 0:
                            await self._close_position(symbol, 'Signal turned positive')
                        
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            logger.error("Stack trace:", exc_info=True)  # Log full stack trace
            
            # Return neutral signal
            return SignalResult(
                combined_signal=0.0,
                confidence=0.0,
                position_size=0.0
            )
            
    def _log_signal_analysis(self, symbol: str, price: float, signal):
        """Log formatted signal analysis."""
        logger.info(f"\nSignal Analysis for {symbol} at {datetime.now(timezone.utc)}:")
        logger.info(f"Current Price: ${price:,.2f}")
        
        # Skip detailed analysis if signals are not valid
        if not (signal.trend_signal and signal.momentum_signal and signal.volatility_signal):
            logger.info("\nNo valid signals - skipping analysis")
            return
            
        # Component signals
        logger.info("\nComponent Signals:")
        # Trend
        logger.info("1. Trend Analysis:")
        logger.info(f"   • ADX: {signal.trend_signal.adx_strength:.1f} ({'Strong' if signal.trend_signal.is_strong_trend else 'Weak'} trend)")
        logger.info(f"   • DMI: {signal.trend_signal.dmi_signal:.3f} ({'Up' if signal.trend_signal.dmi_signal > 0 else 'Down'}trend)")
        logger.info(f"   • MACD: {signal.trend_signal.macd_signal:.3f}")
        logger.info(f"   → Combined: {signal.trend_signal.normalized_signal:.3f} (conf: {signal.trend_signal.confidence:.2f})")
        
        # Momentum
        logger.info("\n2. Momentum Analysis:")
        logger.info(f"   • RSI: {signal.momentum_signal.rsi_value:.1f}")
        logger.info(f"   • StochRSI: {signal.momentum_signal.stoch_rsi:.3f}")
        logger.info(f"   • CCI: {signal.momentum_signal.cci_value:.1f}")
        logger.info(f"   → Combined: {signal.momentum_signal.normalized_signal:.3f} (conf: {signal.momentum_signal.confidence:.2f})")
        
        # Volatility
        logger.info("\n3. Volatility Analysis:")
        logger.info(f"   • BB Width: {signal.volatility_signal.bb_width:.3f}")
        logger.info(f"   • ATR: {signal.volatility_signal.atr_value:.2f}")
        logger.info(f"   • Squeeze: {signal.volatility_signal.squeeze_strength:.3f}")
        logger.info(f"   → Combined: {signal.volatility_signal.normalized_signal:.3f} (conf: {signal.volatility_signal.confidence:.2f})")
        
        # Final analysis
        logger.info("\nFinal Analysis:")
        logger.info(f"Ensemble Signal: {signal.combined_signal:.3f}")
        logger.info(f"Overall Confidence: {signal.confidence:.3f}")
        
        if signal.position_size > 0:
            logger.info("\nTrade Parameters:")
            logger.info(f"Position Size: {signal.position_size:.4f}")
            if signal.stop_loss and signal.take_profit:
                logger.info(f"Stop Loss: ${signal.stop_loss:,.2f}")
                logger.info(f"Take Profit: ${signal.take_profit:,.2f}")
            
    async def _enter_position(self, symbol: str, side: str, params: Dict):
        """Enter a new position."""
        try:
            # Format symbol for Alpaca
            alpaca_symbol = symbol.replace('/', '')
            
            # Create order based on side
            order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=params['position_size'],
                side=order_side,
                time_in_force=TimeInForce.GTC
            )
            
            # Log trade entry attempt
            logger.info(f"\nEntering {side.upper()} position in {symbol}:")
            logger.info(f"Size: {params['position_size']:.4f}")
            logger.info(f"Stop Loss: ${params['stop_loss']:,.2f}")
            logger.info(f"Take Profit: ${params['take_profit']:,.2f}")
            
            # Log order details before submission
            logger.info(f"\nSubmitting {side.upper()} order for {symbol}:")
            logger.info(f"Symbol: {alpaca_symbol}")
            logger.info(f"Quantity: {params['position_size']}")
            logger.info(f"Side: {order_side}")
            
            try:
                # Submit order
                order_result = self.trading_client.submit_order(order)
                logger.info(f"Order submitted successfully. ID: {order_result.id}")
                
                # Track order
                self.pending_orders[alpaca_symbol] = {
                    'order_id': order_result.id,
                    'side': side,
                    'params': params
                }
            except Exception as e:
                logger.error(f"Failed to submit order: {str(e)}")
                return
            
            # Wait for fill
            filled = False
            max_attempts = 5
            attempts = 0
            
            while not filled and attempts < max_attempts:
                try:
                    order_status = self.trading_client.get_order_by_id(order_result.id)
                    if order_status.status == 'filled':
                        # Update position tracking
                        self.current_positions[symbol] = {
                            'side': side,
                            'size': params['position_size'],
                            'entry_price': float(order_status.filled_avg_price),
                            'stop_loss': params['stop_loss'],
                            'take_profit': params['take_profit'],
                            'order_id': order_status.id
                        }
                        logger.info(f"Position opened at ${float(order_status.filled_avg_price):,.2f}")
                        filled = True
                    elif order_status.status in ['canceled', 'expired', 'rejected']:
                        logger.info(f"Order {order_status.status}: {order_status.id}")
                        break
                    else:
                        await asyncio.sleep(1)
                        attempts += 1
                except Exception as e:
                    logger.error(f"Error checking order status: {str(e)}")
                    break
                    
            # Cleanup pending order
            if alpaca_symbol in self.pending_orders:
                del self.pending_orders[alpaca_symbol]
            
        except Exception as e:
            logger.error(f"Error entering position: {str(e)}")
            
    async def _close_position(self, symbol: str, reason: str):
        """Close an existing position."""
        try:
            position = self.current_positions[symbol]
            alpaca_symbol = symbol.replace('/', '')
            
            # Log closing attempt
            logger.info(f"\nClosing {position['side'].upper()} position in {symbol}:")
            logger.info(f"Size: {position['size']:.4f}")
            logger.info(f"Reason: {reason}")
            
            # Create closing order (sell to close long, buy to close short)
            order_side = OrderSide.SELL if position['side'] == 'long' else OrderSide.BUY
            order = MarketOrderRequest(
                symbol=alpaca_symbol,
                qty=position['size'],
                side=order_side,
                time_in_force=TimeInForce.GTC
            )
            
            # Submit order
            order_result = self.trading_client.submit_order(order)
            
            # Wait for fill
            filled = False
            max_attempts = 5
            attempts = 0
            
            while not filled and attempts < max_attempts:
                try:
                    order_status = self.trading_client.get_order_by_id(order_result.id)
                    if order_status.status == 'filled':
                        logger.info(f"Position closed at ${float(order_status.filled_avg_price):,.2f}")
                        # Remove position tracking
                        del self.current_positions[symbol]
                        filled = True
                    elif order_status.status in ['canceled', 'expired', 'rejected']:
                        logger.info(f"Close order {order_status.status}: {order_status.id}")
                        break
                    else:
                        await asyncio.sleep(1)
                        attempts += 1
                except Exception as e:
                    logger.error(f"Error checking close order status: {str(e)}")
                    break
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            
    async def _update_positions(self):
        """Update tracking of current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            
            self.current_positions = {
                pos.symbol: {
                    'side': 'long' if float(pos.qty) > 0 else 'short',
                    'size': abs(float(pos.qty)),
                    'entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pl': float(pos.unrealized_pl)
                }
                for pos in positions
            }
            
        except Exception as e:
            logger.error(f"Error updating positions: {str(e)}")

    async def _handle_websocket(self, websocket):
        """Handle websocket messages."""
        try:
            # Process messages
            async for message in websocket:
                if not self._running:
                    break
                    
                try:
                    data = json.loads(message)
                    if isinstance(data, list):
                        for msg in data:
                            # Handle different message types based on v1beta3 format
                            msg_type = msg.get('T')
                            
                            if msg_type == 'success':
                                logger.info(f"Success message: {msg}")
                            elif msg_type == 'error':
                                logger.error(f"Error message: {msg}")
                            elif msg_type == 'subscription':
                                logger.info(f"Subscription update: {msg}")
                            elif msg_type in ['t', 'q', 'b', 'o']:  # Trade, Quote, Bar, Orderbook
                                await self._handle_market_data(msg)
                    else:
                        # Single message
                        msg_type = data.get('T')
                        if msg_type in ['t', 'q', 'b', 'o']:
                            await self._handle_market_data(data)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON message received")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in websocket handler: {str(e)}")
            
    async def run(self):
        """Main trading loop."""
        # Use v1beta3 crypto websocket URL
        uri = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
        if os.getenv('ALPACA_ENV') == 'paper':
            uri = "wss://stream.data.sandbox.alpaca.markets/v1beta3/crypto/us"
        
        try:
            # Initialize data
            await self.initialize()
            
            logger.info("Starting websocket stream...")
            self._running = True
            
            while self._running:
                try:
                    # Clean up any existing connection
                    if self._ws:
                        try:
                            await self._ws.close()
                            await asyncio.sleep(1)  # Wait for cleanup
                        except:
                            pass
                        self._ws = None
                    
                    # Connect with proper settings
                    async with websockets.connect(
                        uri,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=20,
                        compression=None  # Disable compression
                    ) as websocket:
                        self._ws = websocket
                        
                        try:
                            # Wait for connection success
                            response = await websocket.recv()
                            logger.info(f"Connection response: {response}")
                            
                            # Send authentication with correct v1beta3 format
                            auth_msg = {
                                "action": "auth",
                                "key": self.api_key,
                                "secret": self.secret_key
                            }
                            logger.info("Sending auth message...")
                            await websocket.send(json.dumps(auth_msg))
                            
                            # Wait for auth response
                            logger.info("Waiting for auth response...")
                            response = await websocket.recv()
                            logger.info(f"Raw auth response: {response}")
                            auth_response = json.loads(response)
                        except Exception as e:
                            logger.error(f"Error during websocket setup: {str(e)}")
                            raise
                        
                        if isinstance(auth_response, list) and len(auth_response) > 0 and auth_response[0].get('msg') == 'authenticated':
                            logger.info("Successfully authenticated")
                            
                            # Subscribe to crypto streams with correct v1beta3 format
                            subscribe_msg = {
                                "action": "subscribe",
                                "trades": self.symbols,
                                "quotes": self.symbols,
                                "bars": self.symbols
                            }
                            await websocket.send(json.dumps(subscribe_msg))
                            
                            # Wait for subscription response
                            response = await websocket.recv()
                            logger.info(f"Subscription confirmed: {response}")
                            
                            # Handle messages
                            await self._handle_websocket(websocket)
                        else:
                            logger.error(f"Authentication failed: {response}")
                            logger.error("Too many connections - waiting 30 seconds before retry")
                            await asyncio.sleep(30)  # Longer delay to allow connections to timeout
                            continue
                            
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed, reconnecting...")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"WebSocket error: {str(e)}")
                    await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            raise
        finally:
            await self.stop()
            
    async def stop(self):
        """Stop the trader and cleanup."""
        self._running = False
        if self._ws:
            await self._ws.close()
        logger.info("Trader stopped")
