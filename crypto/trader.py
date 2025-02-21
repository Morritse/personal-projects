import logging
import time
from datetime import datetime, timedelta
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from indicators import get_trading_signals
import config

# Set up logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class CryptoTrader:
    def __init__(self):
        """Initialize the crypto trader with API clients and configuration"""
        self.trading_client = TradingClient(config.API_KEY, config.API_SECRET, paper=True)
        self.data_client = CryptoHistoricalDataClient(config.API_KEY, config.API_SECRET)
        self.positions = {}
        self.initialize_timeframe()
        logger.info("CryptoTrader initialized")

    def initialize_timeframe(self):
        """Initialize timeframe from config"""
        try:
            # Log the timeframe configuration
            logger.info(f"Initializing timeframe with amount={config.TIMEFRAME_AMOUNT}, unit={config.TIMEFRAME_UNIT}")
             
            # Map string to TimeFrameUnit enum
            unit_map = {
                "minute": TimeFrameUnit.Minute,
                "hour": TimeFrameUnit.Hour,
                "day": TimeFrameUnit.Day
            }
            unit = unit_map.get(config.TIMEFRAME_UNIT.lower())
            if unit is None:
                raise ValueError(f"Invalid timeframe unit: {config.TIMEFRAME_UNIT}")
            
            # Create TimeFrame with enum value
            self.timeframe = TimeFrame(amount=config.TIMEFRAME_AMOUNT, unit=unit)
            
            logger.info(f"Timeframe initialized: {self.timeframe}")
        except Exception as e:
            logger.error(f"Error initializing timeframe: {str(e)}")
            logger.warning("Falling back to default timeframe (15 Min)")
            self.timeframe = TimeFrame(amount=15, unit=TimeFrameUnit.Minute)

    def get_historical_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch historical data for a symbol
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            start_dt = datetime.now() - timedelta(days=5)
            end_dt = datetime.now()
            
            logger.info(f"Fetching data for {symbol}")
            logger.info(f"Timeframe: {self.timeframe}")
            logger.info(f"Start: {start_dt.isoformat()}")
            logger.info(f"End: {end_dt.isoformat()}")
            
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self.timeframe,
                start=start_dt,
                end=end_dt
            )
            
            logger.info(f"Request parameters: {request.__dict__}")
            
            # Get the bars data
            bars = self.data_client.get_crypto_bars(request)
            
            # Convert to dataframe
            if hasattr(bars, 'df'):
                df = bars.df
                
                if df is None or df.empty:
                    logger.warning(f"Empty dataframe received for {symbol}")
                    return None
                
                # If the data is for multiple symbols, get just this symbol's data
                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol)
                
                # Reset index to make timestamp a column
                df = df.reset_index()
                
                # Ensure we have required columns for technical analysis
                required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_columns):
                    logger.error(f"Missing required columns. Available columns: {df.columns.tolist()}")
                    return None
                
                # Sort by timestamp to ensure correct order for technical analysis
                df = df.sort_values('timestamp')
                
                logger.info(f"Successfully processed {len(df)} bars for {symbol}")
                return df
            else:
                logger.error(f"Unexpected response format: {type(bars)}")
                return None
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}. Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            return None

    def calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on available buying power
        
        Args:
            price: Current asset price
        
        Returns:
            Position size in USD
        """
        account = self.trading_client.get_account()
        buying_power = float(account.buying_power)
        position_size = min(buying_power * 0.1, config.MAX_POSITION_SIZE)  # Use 10% of buying power
        return position_size

    def place_order(self, symbol: str, side: OrderSide, size: float):
        """
        Place a market order
        
        Args:
            symbol: Trading pair symbol
            side: OrderSide.BUY or OrderSide.SELL
            size: Position size in USD
        """
        try:
            # For crypto, we use notional (USD amount) instead of quantity
            order_request = MarketOrderRequest(
                symbol=symbol,
                notional=size,  # Amount in USD
                side=side,
                time_in_force=TimeInForce.IOC
            )
            order = self.trading_client.submit_order(order_request)
            logger.info(f"Placed {side.name} order for {symbol}: {size} USD")
            return order
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {str(e)}")
            return None

    def manage_positions(self):
        """Update and manage existing positions"""
        try:
            positions = self.trading_client.get_all_positions()
            if positions:
                logger.info("\nManaging existing positions:")
            
            for position in positions:
                symbol = position.symbol
                entry_price = float(position.avg_entry_price)
                current_price = float(position.current_price)
                unrealized_pl_pct = float(position.unrealized_pl_pct)
                qty = float(position.qty)

                logger.info(f"\nPosition: {symbol}")
                logger.info(f"Quantity: {qty:.8f}")
                logger.info(f"Entry Price: ${entry_price:.2f}")
                logger.info(f"Current Price: ${current_price:.2f}")
                logger.info(f"Unrealized P/L: {unrealized_pl_pct:.2%}")

                # Check stop loss and take profit
                if unrealized_pl_pct <= -config.STOP_LOSS_PCT:
                    logger.info(f"🔴 Stop loss triggered for {symbol} at {unrealized_pl_pct:.2%} loss")
                    self.place_order(symbol, OrderSide.SELL, qty)
                elif unrealized_pl_pct >= config.TAKE_PROFIT_PCT:
                    logger.info(f"🟢 Take profit triggered for {symbol} at {unrealized_pl_pct:.2%} profit")
                    self.place_order(symbol, OrderSide.SELL, qty)
                else:
                    logger.info(f"Position within bounds, holding {symbol}")

        except Exception as e:
            logger.error(f"Error managing positions: {str(e)}")

    def run_trading_strategy(self):
        """Main trading loop"""
        while True:
            try:
                # Manage existing positions
                self.manage_positions()

                # Check for new trading opportunities
                positions = self.trading_client.get_all_positions()
                if len(positions) >= config.MAX_POSITIONS:
                    logger.info("Maximum positions reached")
                    time.sleep(60)
                    continue

                for symbol in config.SYMBOLS:
                    # Skip if we already have a position in this symbol
                    if any(p.symbol == symbol for p in positions):
                        continue

                    # Get historical data and generate signals
                    df = self.get_historical_data(symbol)
                    if df is None:
                        continue

                    df = get_trading_signals(df, vars(config))
                    
                    # Check latest signals
                    latest = df.iloc[-1]
                    
                    # Log current market conditions
                    logger.info(f"\nAnalyzing {symbol} at {latest['timestamp']}")
                    logger.info(f"Price: {latest['close']:.2f}")
                    logger.info(f"RSI: {latest['rsi']:.2f} (Need < {config.RSI_OVERSOLD} for buy)")
                    logger.info(f"MACD Histogram: {latest['macd_hist']:.6f} (Need > 0 for buy)")
                    logger.info(f"Volume: {latest['volume']:.6f} vs SMA: {latest['volume_sma']:.6f}")
                    
                    # Check individual conditions
                    rsi_condition = latest['rsi'] < config.RSI_OVERSOLD
                    macd_condition = latest['macd_hist'] > 0
                    volume_condition = latest['volume'] > latest['volume_sma']
                    
                    logger.info("Signal conditions:")
                    logger.info(f"✓ RSI oversold: {rsi_condition}")
                    logger.info(f"✓ MACD positive: {macd_condition}")
                    logger.info(f"✓ Volume > SMA: {volume_condition}")
                    
                    if latest['buy_signal']:
                        logger.info(f"🟢 BUY SIGNAL triggered for {symbol}")
                        position_size = self.calculate_position_size(latest['close'])
                        logger.info(f"Attempting to buy {position_size:.2f} USD worth of {symbol}")
                        self.place_order(symbol, OrderSide.BUY, position_size)
                    else:
                        logger.info(f"⚪ No trade signals for {symbol} - waiting for all conditions to align")
                    
                    time.sleep(1)  # Rate limiting

                # Sleep before next iteration
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)

if __name__ == "__main__":
    trader = CryptoTrader()
    trader.run_trading_strategy()
