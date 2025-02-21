# main.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import logging
import requests

from get_data import CryptoDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoAIAnalyzer:
    def __init__(self):
        self.data_collector = CryptoDataCollector()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100,           # Number of trees in the forest
            max_depth=10,               # Maximum depth of the tree
            min_samples_split=5,        # Minimum number of samples required to split an internal node
            min_samples_leaf=2,         # Minimum number of samples required to be at a leaf node
            random_state=42,
            class_weight='balanced'     # Handle class imbalance
        )
        self.feature_columns = [
            'volume_1h', 'num_transfers',
            'active_whales', 'net_flow',
            'avg_transaction_size', 'whale_buy_pressure'
        ]
        
        # Initialize visualization
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(15, 12))
        self.analysis_history = []
        
        # Initialize price data
        self.price_data = None

    def get_price_data(self, symbol, start_time, end_time):
        """Fetch historical price data from CoinGecko API."""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range"
            params = {
                'vs_currency': 'usd',
                'from': int(start_time),
                'to': int(end_time)
            }
            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"Error fetching price data: HTTP {response.status_code}")
                return pd.DataFrame()
            data = response.json()
            prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
            prices.set_index('timestamp', inplace=True)
            return prices
        except Exception as e:
            logger.error(f"Error fetching price data: {str(e)}")
            return pd.DataFrame()

    def train_model(self, training_period_days=1):
        """Train the model on historical data."""
        logger.info("Collecting historical transaction data...")
        historical_data_usdc = self.data_collector.get_historical_transfers('USDC', days=training_period_days)

        if historical_data_usdc.empty:
            raise ValueError("No historical transaction data available for training.")

        logger.info(f"Preparing features from {len(historical_data_usdc)} historical records...")

        # Fetch historical price data aligned with transaction timestamps
        start_time = historical_data_usdc['timestamp'].min() - 86400  # Extra buffer
        end_time = historical_data_usdc['timestamp'].max() + 86400
        self.price_data = self.get_price_data('usd-coin', start_time, end_time)  # 'usd-coin' for USDC

        if self.price_data.empty:
            raise ValueError("Failed to fetch historical price data.")

        # Resample price data to hourly frequency
        self.price_data = self.price_data.resample('H').last()

        # Merge transaction data with price data
        historical_data_usdc['datetime'] = pd.to_datetime(historical_data_usdc['timestamp'], unit='s')
        historical_data_usdc.set_index('datetime', inplace=True)
        grouped = historical_data_usdc.groupby(pd.Grouper(freq='H')).agg({
            'amount_usd': ['sum', 'mean'],
            'from': 'count',
            'to': 'count'
        })
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

        # Prepare features DataFrame
        features = pd.DataFrame({
            'volume_1h': grouped['amount_usd_sum'],
            'num_transfers': grouped['from_count'],
            'active_whales': grouped['to_count'],  # Number of whale transactions
            'net_flow': grouped['amount_usd_sum'],
            'avg_transaction_size': grouped['amount_usd_mean'].fillna(0),
            'whale_buy_pressure': grouped['to_count'] / grouped['from_count']  # Ratio of whale transactions
        }).fillna(0)

        # Generate labels based on future price movement
        logger.info("Generating training labels based on price changes...")
        price_changes = self.price_data['price'].pct_change().shift(-1)
        labels = (price_changes > 0).astype(int)

        # Align features with labels
        features = features.join(labels, how='inner').dropna()
        labels = labels.loc[features.index]

        if features.empty or labels.empty:
            raise ValueError("Insufficient data after aligning features with labels.")

        logger.info("\nFeature Summary Statistics:")
        logger.info(features.describe())

        # Scale features
        features_scaled = self.scaler.fit_transform(features[self.feature_columns])

        logger.info("Training Random Forest model...")
        self.model.fit(features_scaled, labels)

        # Calculate and log training metrics
        training_score = self.model.score(features_scaled, labels)
        logger.info(f"\nTraining complete:")
        logger.info(f"Training accuracy: {training_score:.2f}")

        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nFeature Importance:")
        for _, row in feature_importance.iterrows():
            logger.info(f"{row['feature']}: {row['importance']:.4f}")

    def prepare_features(self, raw_data):
        """Convert raw transaction data into ML features."""
        current_timestamp = int(datetime.now().timestamp())

        window_start = current_timestamp - 3600  # Last hour
        window_end = current_timestamp

        window = raw_data[
            (raw_data['timestamp'] > window_start) & 
            (raw_data['timestamp'] <= window_end)
        ]

        features = {
            'volume_1h': window['amount_usd'].sum(),
            'num_transfers': len(window),
            'active_whales': len(
                set(window['from']).union(set(window['to']))
                .intersection(self.data_collector.whale_addresses)
            ),
            'net_flow': window['amount_usd'].sum(),
            'avg_transaction_size': window['amount_usd'].mean() if len(window) > 0 else 0,
            'whale_buy_pressure': len(window[
                window['to'].isin(self.data_collector.whale_addresses)
            ]) / max(len(window), 1)
        }

        return pd.DataFrame([features])

    def analyze_current_market(self):
        """Generate trading signals based on current market conditions."""
        recent_data = self.data_collector.transaction_history

        logger.debug("Recent Data Snapshot:")
        logger.debug(recent_data.head())

        if recent_data.empty:
            logger.warning("No recent transaction data available for analysis.")
            return None

        if 'timestamp' not in recent_data.columns:
            logger.error("'timestamp' column not found in recent data.")
            return None

        # Prepare features from recent data
        current_features = self.prepare_features(recent_data)
        if current_features.empty:
            logger.warning("No features could be generated from the recent data.")
            return None

        logger.debug("Generated Features from Recent Data:")
        logger.debug(current_features.head())

        # Scale features and make predictions
        try:
            scaled_features = self.scaler.transform(current_features[self.feature_columns])
            prediction = self.model.predict(scaled_features)
            confidence = self.model.predict_proba(scaled_features)

            # Placeholder for whale analysis
            whale_analysis = {
                'sentiment': 'neutral',  # Replace with actual sentiment analysis
                'risk_level': 'low'      # Replace with actual risk assessment
            }

            # Store analysis for visualization
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'prediction': prediction[0],
                'confidence': confidence[0].max(),
                'whale_sentiment': whale_analysis['sentiment'],
                'risk_level': whale_analysis['risk_level'],
                'volume': recent_data['amount_usd'].sum() if len(recent_data) > 0 else 0
            })

            # Keep last 100 data points
            if len(self.analysis_history) > 100:
                self.analysis_history.pop(0)

            return {
                'prediction': prediction[0],
                'confidence': confidence[0].max(),
                'whale_sentiment': whale_analysis['sentiment'],
                'risk_level': whale_analysis['risk_level']
            }

        except Exception as e:
            logger.error(f"Error during feature scaling or model prediction: {str(e)}")
            return None

    def update_visualization(self):
        """Update the real-time visualization."""
        if not self.analysis_history:
            logger.info("No data to plot yet.")
            return

        history_df = pd.DataFrame(self.analysis_history)

        # Ensure 'timestamp' is in datetime format
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        # Plot 1: Confidence levels over time
        self.ax1.plot(history_df['timestamp'], history_df['confidence'], 
                      color='cyan', linewidth=2)
        self.ax1.set_title('Model Confidence Over Time')
        self.ax1.set_ylabel('Confidence')

        # Plot 2: Whale Sentiment
        whale_colors = {
            'strongly_bullish': 'green',
            'bullish': 'lightgreen',
            'neutral': 'yellow',
            'bearish': 'orange',
            'strongly_bearish': 'red'
        }

        sentiment_colors = [whale_colors.get(s, 'gray') for s in history_df['whale_sentiment']]
        self.ax2.scatter(history_df['timestamp'], 
                        [1]*len(history_df), 
                        c=sentiment_colors, 
                        s=100)
        self.ax2.set_title('Whale Sentiment')
        self.ax2.set_yticks([])

        # Plot 3: Volume and Activity
        self.ax3.bar(history_df['timestamp'], 
                    history_df['volume'], 
                    alpha=0.6, 
                    color='blue')
        self.ax3.set_title('Trading Volume')
        self.ax3.set_ylabel('Volume (USD)')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)

    def backtest_model(self, backtest_days=7):
        """Optional: Implement backtesting to evaluate model performance."""
        pass  # Placeholder for backtesting implementation

def main():
    analyzer = CryptoAIAnalyzer()
    
    try:
        logger.info("Training model...")
        # Use a smaller training period for development
        analyzer.train_model(training_period_days=1)
    except Exception as e:
        logger.error(f"Failed to train the model: {str(e)}")
        return
    
    plt.ion()  # Enable interactive plotting
    plt.show()  # Ensure the plot window is displayed

    logger.info("\nStarting real-time analysis...")
    logger.info("Press Ctrl+C to stop")

    while True:
        try:
            analyzer.data_collector.collect_real_time_data()  # Update transaction history
            analysis = analyzer.analyze_current_market()

            if analysis:
                prediction_str = 'BUY' if analysis['prediction'] == 1 else 'SELL'
                logger.info("\nMarket Analysis:")
                logger.info(f"Prediction: {prediction_str}")
                logger.info(f"Confidence: {analysis['confidence']:.2f}")
                logger.info(f"Whale Sentiment: {analysis['whale_sentiment']}")
                logger.info(f"Risk Level: {analysis['risk_level']}")

            analyzer.update_visualization()

            time.sleep(300)  # 5-minute update interval

        except KeyboardInterrupt:
            logger.info("\nStopping analysis...")
            break
        except Exception as e:
            logger.error(f"Error in analysis loop: {str(e)}")
            time.sleep(60)
    
    plt.ioff()
    plt.close()

if __name__ == "__main__":
    main()
