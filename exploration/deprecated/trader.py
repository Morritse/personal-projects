from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_PAPER,
    ALPACA_PAPER_URL,
    ALPACA_DATA_URL,
    VERBOSE_DATA
)

class AlpacaTrader:
    def __init__(self):
        if VERBOSE_DATA:
            print("\nInitializing Alpaca Trading Client...")
            print(f"Base URL: {ALPACA_PAPER_URL}")
            
        # Trading client for orders and positions
        self.client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            paper=ALPACA_PAPER,
            url_override=ALPACA_PAPER_URL
        )
        
        # Data client for latest prices
        self.data_client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            url_override=ALPACA_DATA_URL
        )
        
        # Get account information
        account = self.client.get_account()
        if VERBOSE_DATA:
            print(f"Account ready for trading. Cash: ${float(account.cash):.2f}")

    def get_position(self, symbol: str) -> float:
        """Get current position size for a symbol. Returns positive for long, negative for short."""
        try:
            position = self.client.get_position(symbol)
            return float(position.qty)
        except:
            return 0

    def close_position(self, symbol: str) -> bool:
        """Close any existing position for a symbol."""
        try:
            self.client.close_position(symbol)
            return True
        except Exception as e:
            if VERBOSE_DATA:
                print(f"Error closing position for {symbol}: {str(e)}")
            return False

    def submit_order(self, symbol: str, side: str, signal_strength: float = 0.25) -> bool:
        """Submit a market order."""
        try:
            # Calculate position size based on signal strength
            account = self.client.get_account()
            buying_power = float(account.buying_power)
            
            # Scale position size from 3% to 10% based on signal strength
            # Signal strength ranges: WEAK = 0.25, STRONG = 0.5
            signal_strength = abs(float(signal_strength))
            base_size = 0.03  # 3% minimum
            max_size = 0.10   # 10% maximum
            
            # Linear scaling between base_size and max_size
            scale_factor = min(1.0, max(0, (signal_strength - 0.25) / (0.5 - 0.25)))
            position_size_pct = base_size + (max_size - base_size) * scale_factor
            
            # Calculate position value
            position_value = buying_power * position_size_pct
            
            # Get current price from latest trade
            trade_request = StockLatestTradeRequest(
                symbol_or_symbols=symbol,
                feed='sip'  # Use SIP feed for better data quality
            )
            trade = self.data_client.get_stock_latest_trade(trade_request)
            price = float(trade[symbol].price)
            
            # Calculate quantity
            qty = int(position_value / price)
            if qty < 1:
                if VERBOSE_DATA:
                    print(f"Insufficient funds for {symbol} position")
                return False
            
            # Create market order
            order_data = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY if side == "BUY" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = self.client.submit_order(order_data)
            if VERBOSE_DATA:
                print(f"Order submitted for {symbol}: {side} {qty} shares")
            return True
            
        except Exception as e:
            if VERBOSE_DATA:
                print(f"Error submitting order for {symbol}: {str(e)}")
            return False

    def get_account_info(self) -> dict:
        """Get current account information."""
        try:
            account = self.client.get_account()
            return {
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power)
            }
        except Exception as e:
            if VERBOSE_DATA:
                print(f"Error getting account info: {str(e)}")
            return {
                "cash": 0.0,
                "equity": 0.0,
                "buying_power": 0.0
            }
