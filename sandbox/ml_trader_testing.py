import os
import time
import logging
import pickle
from datetime import datetime, timezone, timedelta  # Make sure timedelta is imported
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from dateutil.parser import isoparse
import yfinance as yf
import traceback
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
        
        # Add Indicators configuration
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
        
        # State variables initialization
        self.position = None
        self.buy_order_id = None
        self.last_trade_time = None
        self.trades = []
        self.buy_price = None
        self.trailing_stop_price = None
        
        # Initialize data with historical data
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

            self.data_with_indicators = self.alpaca_utils.calculate_indicators(
                initial_data,
                self.INDICATORS_CONFIG
            )

            self.logger.info("Calculated indicators for initial data")
            
        except Exception as e:
            self.logger.error(f"Error initializing data: {e}")
            # Create empty DataFrame with required columns
            self.data_with_indicators = pd.DataFrame(
                columns=['Datetime', 'open', 'high', 'low', 'close', 'volume']
            )
            
        # Setup visualization
        self.setup_visualization()
    
    def test_with_historical_data(self, num_samples=100):
        """Test the model with historical data."""
        self.logger.info("\n=== Testing with Historical Data ===")
        try:
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)  # Adjust as needed
            historical_data = yf.download(self.symbol, start=start_date, end=end_date, interval='1m')
            historical_data.reset_index(inplace=True)
            historical_data.rename(columns={
                'Datetime': 'Datetime',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            # Calculate indicators
            data_with_indicators = self.alpaca_utils.calculate_indicators(historical_data, self.INDICATORS_CONFIG)


            # Ensure we have enough data
            if len(data_with_indicators) < num_samples:
                self.logger.warning(f"Not enough data for testing. Required: {num_samples}, Available: {len(data_with_indicators)}")
                return None

            # Prepare features
            features = data_with_indicators[self.feature_columns].tail(num_samples).values
            features = self.scaler.transform(features)

            # Get model predictions
            rf_preds = self.rf_model.predict(features)
            xgb_preds = self.xgb_model.predict(features)

            # Ensemble predictions
            combined_preds = (rf_preds + xgb_preds) / 2
            final_preds = (combined_preds > 0.5).astype(int)

            # Actual targets
            actuals = data_with_indicators['close'].shift(-5).tail(num_samples).values > data_with_indicators['close'].tail(num_samples).values
            actuals = actuals.astype(int)

            # Calculate accuracy
            accuracy = np.mean(final_preds == actuals)
            self.logger.info(f"Historical Data Test Accuracy: {accuracy:.4f}")

            return {
                'accuracy': accuracy,
                'predictions': final_preds,
                'actuals': actuals
            }

        except Exception as e:
            self.logger.error(f"Error during historical data testing: {e}")
            self.logger.error(traceback.format_exc())
            return None


    def validate_setup(self):
        """Validate models, data, and features after initialization"""
        self.logger.info("\n=== Validating Trading Setup ===")
        
        # Check models loaded correctly
        self.logger.info("\nModel Information:")
        self.logger.info(f"RandomForest Features: {len(self.rf_model.feature_importances_)}")
        # For XGBoost, use feature_names instead
        self.logger.info(f"XGBoost Features: {len(self.feature_columns)}")
        
        # Print feature importance for RF
        rf_importances = pd.Series(
            self.rf_model.feature_importances_,
            index=self.feature_columns
        ).sort_values(ascending=False)
        
        # Print feature importance for XGBoost
        self.logger.info("\nTop 5 Important Features (RandomForest):")
        for feat, imp in rf_importances.head().items():
            self.logger.info(f"{feat}: {imp:.4f}")
        
        # Try to get XGBoost feature importance if available
        try:
            xgb_importance = self.xgb_model.get_booster().get_score(importance_type='weight')
            xgb_importance = pd.Series(xgb_importance).sort_values(ascending=False)
            self.logger.info("\nTop 5 Important Features (XGBoost):")
            for feat, imp in xgb_importance.head().items():
                self.logger.info(f"{feat}: {imp:.4f}")
        except:
            self.logger.info("Could not get XGBoost feature importance")
        
        # Test prediction pipeline
        self.logger.info("\nTesting Prediction Pipeline:")
        dummy_data = pd.DataFrame({
            col: [0.0] for col in self.feature_columns
        })
        features = self.prepare_features(dummy_data)
        buy_signal, sell_signal, confidence = self.get_model_predictions(features)
        self.logger.info(
            f"Test Prediction Result - Buy: {buy_signal}, "
            f"Sell: {sell_signal}, Confidence: {confidence:.2f}"
        )
        
        # Validate features
        self.logger.info("\nFeature Validation:")
        self.logger.info(f"Expected Features: {self.feature_columns}")
        missing_features = [f for f in self.feature_columns 
                        if f not in self.data_with_indicators.columns]
        self.logger.info(f"Missing Features: {missing_features if missing_features else 'None'}")
        
        return not bool(missing_features)  # Return True if no missing features


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
            
            # Ensemble prediction with detailed logging
            buy_signal = (rf_pred == 1 and xgb_pred == 1 and 
                         rf_prob > 0.6 and xgb_prob > 0.6)
            sell_signal = (rf_pred == 0 and xgb_pred == 0 and 
                          rf_prob < 0.4 and xgb_prob < 0.4)
            
            confidence = (rf_prob + xgb_prob) / 2
            
            # Log detailed prediction info
            self.logger.debug(
                f"Predictions - RF: {rf_pred} ({rf_prob:.2f}), "
                f"XGB: {xgb_pred} ({xgb_prob:.2f})"
            )
            
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
            
        self.update_data(price_data)
        
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
        
        self.ax1.set_title('Price and Trades')
        self.ax2.set_title('Buy Signal Confidence')
        plt.tight_layout()

    def update_data(self, price_data):
            """Update data with latest price information"""
            try:
                latest_price = price_data[self.symbol].get("price")
                latest_timestamp = isoparse(price_data[self.symbol].get("timestamp"))
                
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
                
                self.data_with_indicators = self.alpaca_utils.calculate_indicators(
                    self.data_with_indicators,
                    self.INDICATORS_CONFIG
                )
                
                self.make_trading_decision(self.data_with_indicators)
                
            except Exception as e:
                self.logger.error(f"Error updating data: {e}")

    def run(self):
        """Main run loop"""
        try:
            self.logger.info("Starting trading system...")
            self.client.start()
            
            ani = FuncAnimation(self.fig, self.update_plot, interval=1000)
            plt.ion()
            
        except KeyboardInterrupt:
            self.logger.info("Shutting down gracefully...")
            self.client.close()
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
            self.client.close()

if __name__ == "__main__":
    # Configuration
    SYMBOL = "SPY"
    API_KEY = "YOUR_API_KEY"       # Replace with your Alpaca API key
    API_SECRET = "YOUR_API_SECRET"  # Replace with your Alpaca API secret
    MODEL_PATH = "local_models_SPY.pkl"  # Path to your optimized models
    
    # Initialize trader
    trader = MLTrader(SYMBOL, API_KEY, API_SECRET, MODEL_PATH)
    
    # Run validation checks
    validation_success = trader.validate_setup()
    if not validation_success:
        print("Validation failed. Please check the logs.")
        exit(1)
    
    # Test with historical data
    test_results = trader.test_with_historical_data(num_samples=100)
    if test_results is None:
        print("Historical data test failed. Please check the logs.")
        exit(1)
    
    # Ask for confirmation before starting live trading
    input("\nValidation complete. Press Enter to start live trading...")
    
    # Run trader
    trader.run()
