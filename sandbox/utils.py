import json
import logging
import threading
import time
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import websocket
import ta
from alpaca_trade_api.rest import REST
from matplotlib.animation import FuncAnimation

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AlpacaWebSocketClient:
    def __init__(self, api_key, api_secret, symbols):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.ws_url = (
            "wss://stream.data.alpaca.markets/v2/sip"  # Update to appropriate endpoint
        )
        self.raw_messages = []
        self.price_data = {}
        self.data_lock = threading.Lock()
        self.ws = None

    def on_message(self, ws, message):
        try:
            data = json.loads(message)
            with self.data_lock:
                self.raw_messages.append(data)
                if len(self.raw_messages) > 1000:
                    self.raw_messages.pop(0)
            logging.info(f"Raw message received: {data}")
            self.process_message(data)
        except Exception as e:
            logging.error(f"Error processing WebSocket message: {e}")

    def on_error(self, ws, error):
        logging.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logging.info("WebSocket connection closed.")

    def close(self):
        if self.ws:
            self.ws.close()
            logging.info("WebSocket connection closed by client.")
            
    def on_open(self, ws):
        logging.info("WebSocket connection opened.")
        auth_payload = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret,
        }
        ws.send(json.dumps(auth_payload))
        logging.info("Sent authentication request.")

        # Subscribe to the specified symbols
        subscribe_message = {"action": "subscribe", "trades": self.symbols}
        ws.send(json.dumps(subscribe_message))
        logging.info(f"Subscribed to real-time trades for: {self.symbols}")

    def process_message(self, data):
        with self.data_lock:
            for msg in data:
                msg_type = msg.get("T")
                if msg_type == "t":  # Trade updates
                    symbol = msg.get("S")
                    price = msg.get("p")
                    timestamp = msg.get("t")
                    self.price_data[symbol] = {"price": price, "timestamp": timestamp}
                    logging.info(f"Trade update - {symbol}: ${price} at {timestamp}")
                elif msg_type == "success":
                    logging.info(f"Success message: {msg.get('msg')}")
                elif msg_type == "subscription":
                    logging.info(f"Subscription message: {msg}")
                elif msg_type == "error":
                    logging.error(f"Error message: {msg.get('msg')}")
                else:
                    logging.warning(f"Unhandled message type: {msg}")

    def start(self):
        def run_websocket():
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            self.ws.run_forever()

        threading.Thread(target=run_websocket, daemon=True).start()

    def get_price_data(self):
        with self.data_lock:
            return self.price_data.copy()


class LiveGraph:
    def __init__(self, client, symbols, refresh_interval=1000):
        self.client = client
        self.symbols = symbols
        self.refresh_interval = refresh_interval
        self.prices = {symbol: [] for symbol in symbols}
        self.timestamps = {symbol: [] for symbol in symbols}

        # Set up the plot
        self.fig, self.ax = plt.subplots()
        plt.style.use("seaborn-darkgrid")
        self.ax.set_title("Live Price Data")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")

    def update_plot(self, frame):
        with self.client.data_lock:
            for symbol in self.symbols:
                # Get latest price data
                price_data = self.client.get_price_data()
                if symbol in price_data:
                    latest_price = price_data[symbol]["price"]
                    latest_timestamp = datetime.strptime(
                        price_data[symbol]["timestamp"], "%Y-%m-%dT%H:%M:%S.%fZ"
                    )

                    self.prices[symbol].append(latest_price)
                    self.timestamps[symbol].append(latest_timestamp)

                    # Limit data to the last 100 points
                    if len(self.prices[symbol]) > 100:
                        self.prices[symbol].pop(0)
                        self.timestamps[symbol].pop(0)

            # Clear and redraw the plot
            self.ax.clear()
            for symbol in self.symbols:
                self.ax.plot(
                    self.timestamps[symbol],
                    self.prices[symbol],
                    label=f"{symbol} Price",
                )

            # Format plot
            self.ax.legend()
            self.ax.set_title("Live Price Data")
            self.ax.set_xlabel("Time")
            self.ax.set_ylabel("Price")
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            plt.xticks(rotation=45)
            plt.tight_layout()

    def start(self):
        # Start the live plot animation
        ani = FuncAnimation(self.fig, self.update_plot, interval=self.refresh_interval)
        plt.show()


class AlpacaUtils:
    def __init__(self, api_key, api_secret):
        from alpaca_trade_api import REST

        self.api = REST(
            api_key,
            api_secret,
            base_url="https://paper-api.alpaca.markets",
            api_version="v2",
        )
        self.data_api = REST(
            api_key,
            api_secret,
            base_url="https://data.alpaca.markets",
            api_version="v2",
        )
        logging.info("AlpacaUtils initialized")

    def fetch_historical_data(self, symbol, timeframe="1Min", limit=1000):
        """
        Fetch historical data for a given symbol from Alpaca.
        """
        logging.debug(f"Fetching historical data for {symbol}")
        try:
            bars = self.api.get_bars(symbol, timeframe, limit=limit).df
            if not bars.empty:
                bars.reset_index(inplace=True)
                bars.rename(columns={"timestamp": "Datetime"}, inplace=True)
                bars["Datetime"] = bars["Datetime"].dt.tz_convert(None)
                logging.debug(
                    f"Successfully fetched data for {symbol}: {len(bars)} rows."
                )
                return bars
            else:
                logging.warning(f"No data found for {symbol}")
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
        return None

    def calculate_indicators(self, data, indicators_config):
        """
        Calculates technical indicators for the provided data.

        Parameters:
            data (pd.DataFrame): The historical data for the symbol.
            indicators_config (dict): Configuration for indicators to calculate.

        Returns:
            pd.DataFrame: The original data with calculated indicators.
        """
        try:
            # Ensure the data contains required columns
            required_columns = ["close", "high", "low"]
            if not all(col in data.columns for col in required_columns):
                raise ValueError(
                    f"Data is missing required columns: {required_columns}"
                )

            # Calculate indicators based on configuration
            if "SMA" in indicators_config:
                window = indicators_config["SMA"]["window"]
                data[f"SMA_{window}"] = ta.trend.sma_indicator(
                    data["close"], window=window
                )

            if "EMA" in indicators_config:
                window = indicators_config["EMA"]["window"]
                data[f"EMA_{window}"] = ta.trend.ema_indicator(
                    data["close"], window=window
                )

            if "RSI" in indicators_config:
                window = indicators_config["RSI"]["window"]
                data[f"RSI_{window}"] = ta.momentum.rsi(data["close"], window=window)

            if "MACD" in indicators_config:
                macd = ta.trend.MACD(
                    data["close"],
                    window_slow=indicators_config["MACD"]["slow"],
                    window_fast=indicators_config["MACD"]["fast"],
                    window_sign=indicators_config["MACD"]["signal"],
                )
                data["MACD"] = macd.macd()
                data["MACD_Signal"] = macd.macd_signal()
                data["MACD_Hist"] = macd.macd_diff()

            if "Bollinger" in indicators_config:
                window = indicators_config["Bollinger"]["window"]
                window_dev = indicators_config["Bollinger"]["window_dev"]
                # Calculate rolling mean and std
                rolling_mean = data['close'].rolling(window=window).mean()
                rolling_std = data['close'].rolling(window=window).std()
                data['BB_middle'] = rolling_mean
                data['BB_upper'] = rolling_mean + (rolling_std * window_dev)
                data['BB_lower'] = rolling_mean - (rolling_std * window_dev)

            if "ATR" in indicators_config:
                window = indicators_config["ATR"]["window"]
                data[f"ATR_{window}"] = ta.volatility.average_true_range(
                    data["high"], data["low"], data["close"], window=window
                )

            # **Additional features not previously included**

            # Returns
            data['returns'] = data['close'].pct_change()

            # Returns Volatility
            data['returns_volatility'] = data['returns'].rolling(window=20).std()

            # Drop rows with NaN values resulting from indicator calculations
            data.dropna(inplace=True)

            return data
        except Exception as e:
            logging.error(f"Error calculating indicators: {e}")
            return None


    def fetch_and_calculate_indicators(
        self, symbol, timeframe="1Min", limit=1000, indicators_config=None
    ):
        """
        Fetch historical data and calculate indicators in one step.
        """
        if indicators_config is None:
            indicators_config = {}
        data = self.fetch_historical_data(symbol, timeframe, limit)
        if data is not None:
            return self.calculate_indicators(data, indicators_config)
        else:
            logging.error(
                f"Failed to fetch data for {symbol}. Cannot calculate indicators."
            )
        return None


# Main function for standalone execution
def main():
    # Replace with your actual API keys
    api_key = "YOUR_API_KEY"
    api_secret = "YOUR_API_SECRET"
    symbols = ["AAPL", "MSFT"]  # Replace with desired symbols

    client = AlpacaWebSocketClient(api_key, api_secret, symbols)
    client.start()

    while True:
        time.sleep(1)
        price_data = client.get_price_data()
        for symbol, data in price_data.items():
            print(f"{symbol}: ${data['price']} at {data['timestamp']}")


if __name__ == "__main__":
    main()
