import os
import time
import logging
import pickle
import traceback
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dateutil.parser import isoparse
from utils import AlpacaWebSocketClient, AlpacaUtils

# Setup main logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading.log')
    ]
)

class MLTrader:
    def __init__(self, symbol, api_key, api_secret, model_path):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing ML Trader")
        
        # Trading configuration
        self.symbol = symbol
        self.timeframe = "1Min"
        self.limit = 1000
        self.cooldown_period = timedelta(minutes=1)
        self.position_size = 0.10
        self.max_drawdown = 0.02
        self.trailing_stop = 0.01
        
        # Indicators configuration
        self.INDICATORS_CONFIG = {
            "SMA": {"window": 10},
            "EMA": {"window": 5},
            "RSI": {"window": 14},
            "MACD": {"fast": 12, "slow": 26, "signal": 9},
            "Bollinger": {"window": 20, "window_dev": 2},
            "ATR": {"window": 14},
        }
        
        # Initialize API connections
        self.alpaca_utils = AlpacaUtils(api_key, api_secret)
        self.client = AlpacaWebSocketClient(api_key, api_secret, [symbol])
        
        # Load ML models
        self.load_models(model_path)
        
        # State variables
        self.position = None
        self.buy_order_id = None
        self.last_trade_time = None
        self.trades = []
        self.buy_price = None
        self.trailing_stop_price = None
        
        # Initialize data with historical data
        try:
            self.logger.info("Attempting to fetch data from Alpaca...")
            initial_data = self.alpaca_utils.fetch_historical_data(
                symbol=symbol,
                timeframe=self.timeframe,
                limit=100
            )
            
            if initial_data is None or initial_data.empty:
                self.logger.info("Falling back to yfinance for initial data...")
                import yfinance as yf
                
                end = datetime.now()
                start = end - timedelta(days=5)  # Get 5 days of data
                
                yf_data = yf.download(symbol, start=start, end=end, interval='1m')
                yf_data.reset_index(inplace=True)
                
                # Rename columns to match expected format
                yf_data = yf_data.rename(columns={
                    'Datetime': 'Datetime',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                initial_data = yf_data

            # Calculate indicators
            self.data_with_indicators = self.alpaca_utils.calculate_indicators(
                initial_data,
                self.INDICATORS_CONFIG
            )

            self.logger.info("Calculated indicators for initial data")
            
        except Exception as e:
            self.logger.error(f"Error initializing data: {e}")
            self.logger.error(traceback.format_exc())
            # Create empty DataFrame with required columns
            self.data_with_indicators = pd.DataFrame(
                columns=['Datetime', 'open', 'high', 'low', 'close', 'volume']
            )
            
        # Setup visualization
        self.setup_visualization()

    def load_models(self, model_path):
        """Load optimized ML models"""
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
                
            self.rf_model = self.model_data['rf_model']
            self.xgb_model = self.model_data['xgb_model']
            self.feature_columns = self.model_data['feature_columns']
            self.scaler = self.model_data['scaler']
            
            self.logger.info(f"Successfully loaded models for {self.symbol}")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def prepare_features(self, data):
        """Prepare features for ML prediction"""
        try:
            features = data[self.feature_columns].iloc[-1:].values
            features = self.scaler.transform(features)
            return features
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None

    def get_model_predictions(self, features):
        """Get predictions from both models"""
        try:
            rf_pred = self.rf_model.predict(features)[0]
            rf_prob = self.rf_model.predict_proba(features)[0][1]
            
            xgb_pred = self.xgb_model.predict(features)[0]
            xgb_prob = self.xgb_model.predict_proba(features)[0][1]
            
            # Ensemble prediction
            buy_signal = (rf_pred == 1 and xgb_pred == 1 and 
                         rf_prob > 0.6 and xgb_prob > 0.6)
            sell_signal = (rf_pred == 0 and xgb_pred == 0 and 
                          rf_prob < 0.4 and xgb_prob < 0.4)
            
            confidence = (rf_prob + xgb_prob) / 2
            
            return buy_signal, sell_signal, confidence
            
        except Exception as e:
            self.logger.error(f"Error getting predictions: {e}")
            return False, False, 0

    def check_risk_limits(self, latest_price):
        """Check if we've hit any risk limits"""
        if self.position == 'long' and self.buy_price:
            # Check max drawdown
            current_drawdown = (latest_price - self.buy_price) / self.buy_price
            if current_drawdown < -self.max_drawdown:
                self.logger.info(f"Max drawdown hit: {current_drawdown:.2%}")
                return True
                
            # Update trailing stop
            if self.trailing_stop_price is None:
                self.trailing_stop_price = self.buy_price * (1 - self.trailing_stop)
            elif latest_price > self.buy_price:
                new_stop = latest_price * (1 - self.trailing_stop)
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
                    self.logger.info(f"Updated trailing stop: ${self.trailing_stop_price:.2f}")
                    
            if latest_price < self.trailing_stop_price:
                self.logger.info("Trailing stop hit")
                return True
                
        return False

    def make_trading_decision(self, data):
        """Make trading decisions based on ML predictions and risk management"""
        try:
            self.logger.info("\n=== Making Trading Decision ===")
            
            if len(data) < 20:
                self.logger.info("Not enough data for features")
                return
                
            current_time = datetime.utcnow()
            if self.last_trade_time and (current_time - self.last_trade_time) < self.cooldown_period:
                self.logger.info("In cooldown period")
                return

            features = self.prepare_features(data)
            if features is None:
                return
                
            buy_signal, sell_signal, confidence = self.get_model_predictions(features)
            self.logger.info(
                f"Signals - Buy: {buy_signal}, Sell: {sell_signal}, "
                f"Confidence: {confidence:.2f}"
            )
            
            latest_price = data['close'].iloc[-1]
            risk_limit_hit = self.check_risk_limits(latest_price)
            
            if risk_limit_hit or sell_signal:
                self.execute_sell(latest_price, data['Datetime'].iloc[-1])
            elif buy_signal and self.position is None:
                self.execute_buy(latest_price, data['Datetime'].iloc[-1], confidence)
                
        except Exception as e:
            self.logger.error(f"Error in trading decision: {e}")
            self.logger.error(traceback.format_exc())

    def execute_buy(self, price, timestamp, confidence):
        """Execute buy order with position sizing based on confidence"""
        try:
            account = self.alpaca_utils.api.get_account()
            buying_power = float(account.buying_power)
            
            adjusted_size = self.position_size * confidence
            position_value = buying_power * adjusted_size
            
            qty = int(position_value / price)
            if qty < 1:
                self.logger.info("Order quantity too small")
                return
                
            order_params = {
                'symbol': self.symbol,
                'qty': qty,
                'side': 'buy',
                'type': 'market',
                'time_in_force': 'gtc',
            }
            
            self.logger.info(f"Submitting buy order: {order_params}")
            order = self.submit_order_with_retry(order_params)
            
            if order:
                self.position = 'long'
                self.buy_order_id = order.id
                self.last_trade_time = datetime.utcnow()
                self.buy_price = price
                self.trailing_stop_price = None
                
                self.trades.append({
                    'type': 'buy',
                    'price': price,
                    'datetime': timestamp,
                    'confidence': confidence
                })
                
                self.logger.info(f"Buy order executed: {qty} shares at ${price:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            self.logger.error(traceback.format_exc())

    def execute_sell(self, price, timestamp):
        """Execute sell order"""
        try:
            if self.position != 'long':
                return
                
            positions = self.alpaca_utils.api.list_positions()
            for pos in positions:
                if pos.symbol == self.symbol:
                    order_params = {
                        'symbol': self.symbol,
                        'qty': pos.qty,
                        'side': 'sell',
                        'type': 'market',
                        'time_in_force': 'gtc',
                    }
                    
                    self.logger.info(f"Submitting sell order: {order_params}")
                    order = self.submit_order_with_retry(order_params)
                    
                    if order:
                        self.position = None
                        self.last_trade_time = datetime.utcnow()
                        self.buy_price = None
                        self.trailing_stop_price = None
                        
                        self.trades.append({
                            'type': 'sell',
                            'price': price,
                            'datetime': timestamp
                        })
                        
                        self.logger.info(f"Sell order executed at ${price:.2f}")
                        
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            self.logger.error(traceback.format_exc())

    def submit_order_with_retry(self, order_params, max_retries=3, delay=2):
        """Submit order with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.alpaca_utils.api.submit_order(**order_params)
            except Exception as e:
                self.logger.error(f"Order submission failed on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
        return None

    def setup_visualization(self):
        """Setup matplotlib visualization"""
        plt.style.use("seaborn-darkgrid")
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle(f"{self.symbol} Trading Dashboard")

    def update_plot(self, frame):
        """Update plot with latest data"""
        price_data = self.client.get_price_data()
        if self.symbol not in price_data:
            return
            
        # Update data and make trading decisions
        self.update_data(price_data)
        
        # Update plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Price plot
        self.ax1.plot(self.data_with_indicators['Datetime'], 
                     self.data_with_indicators['close'], 
                     label='Price')
        
        # Plot trades
        for trade in self.trades:
            if trade['type'] == 'buy':
                self.ax1.scatter(trade['datetime'], trade['price'], 
                               color='green', marker='^', s=100)
            else:
                self.ax1.scatter(trade['datetime'], trade['price'], 
                               color='red', marker='v', s=100)
        
        # Model confidence plot
        if self.trades:
            confidence_data = pd.DataFrame(self.trades)
            confidence_data = confidence_data[confidence_data['type'] == 'buy']
            if not confidence_data.empty:
                self.ax2.bar(confidence_data['datetime'], 
                           confidence_data['confidence'], 
                           color='blue', alpha=0.6)
        
        # Formatting
        self.ax1.set_title('Price and Trades')
        self.ax2.set_title('Buy Signal Confidence')
        plt.tight_layout()

    def update_data(self, price_data):
        """Update data with latest price information"""
        try:
            latest_price = price_data[self.symbol].get("price")
            timestamp_str = price_data[self.symbol].get("timestamp")
            if timestamp_str is None:
                self.logger.error("Timestamp not found in price data.")
                return
            latest_timestamp = isoparse(timestamp_str)
            
            new_row = pd.DataFrame({
                'Datetime': [latest_timestamp],
                'open': [latest_price],
                'high': [latest_price],
                'low': [latest_price],
                'close': [latest_price],
                'volume': [0]
            })
            
            self.data_with_indicators = pd.concat(
                [self.data_with_indicators, new_row], 
                ignore_index=True
            )
            
            # Calculate indicators and make trading decision
            self.data_with_indicators = self.alpaca_utils.calculate_indicators(
                self.data_with_indicators,
                self.INDICATORS_CONFIG
            )
            
            self.make_trading_decision(self.data_with_indicators)
            
        except Exception as e:
            self.logger.error(f"Error updating data: {e}")
            self.logger.error(traceback.format_exc())

    def run(self):
        """Main run loop"""
        try:
            self.logger.info("Starting trading system...")
            self.client.start()
            
            # Non-blocking plot
            ani = FuncAnimation(self.fig, self.update_plot, interval=1000)
            plt.show(block=False)
            
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down gracefully...")
            self.client.close()
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.logger.error(traceback.format_exc())
            self.client.close()

if __name__ == "__main__":
    # Configuration
    SYMBOL = "SPY"
    API_KEY = "YOUR_API_KEY"       # Replace with your Alpaca API key
    API_SECRET = "YOUR_API_SECRET"  # Replace with your Alpaca API secret
    MODEL_PATH = "local_models_SPY.pkl"  # Path to your optimized models
    
    # Initialize trader
    trader = MLTrader(SYMBOL, API_KEY, API_SECRET, MODEL_PATH)
    
    # Run trader
    trader.run()
