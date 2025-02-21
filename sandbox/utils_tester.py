import os
import time
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.dates as mdates
from dateutil.parser import isoparse

from utils import AlpacaWebSocketClient, AlpacaUtils  # Ensure these are correctly implemented

# ==================== Configuration ====================

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Alpaca API Credentials (Use environment variables for security)
API_KEY = "YOUR_API_KEY"       # Replace with your Alpaca API key
API_SECRET = "YOUR_API_SECRET"  # Replace with your Alpaca API secret

# Initialize AlpacaUtils and WebSocketClient
alpaca_utils = AlpacaUtils(API_KEY, API_SECRET)
symbols = ["TSLA"]  # Using SPY as it's confirmed working
client = AlpacaWebSocketClient(API_KEY, API_SECRET, symbols)
client.start()

# Trading Configuration
SYMBOL = "TSLA"  # Trading symbol
TIMEFRAME = "1Min"
LIMIT = 1000
INDICATORS_CONFIG = {
    "SMA": {"window": 10},
    "EMA": {"window": 5},
    "RSI": {"window": 14},
    "MACD": {"fast": 12, "slow": 26, "signal": 9},
    "Bollinger": {"window": 20, "window_dev": 2},
    "ATR": {"window": 14},
}

# Matplotlib setup for live graphing
plt.style.use("seaborn-darkgrid")
fig, ax = plt.subplots()
ax.set_title("Live Price Data with Indicators")
ax.set_xlabel("Time")
ax.set_ylabel("Price")

# ==================== Global Variables ====================

data_with_indicators = pd.DataFrame(
    columns=['Datetime', 'open', 'high', 'low', 'close', 'volume']
)
position = None  # Possible values: None (no position), 'long' (holding a long position)
buy_order_id = None  # To track the order ID of the buy order
trades = []  # List to store executed trades

# Cooldown settings to prevent rapid order placements
last_trade_time = None
COOLDOWN_PERIOD = timedelta(minutes=1)  # Example: 1-minute cooldown

# ==================== Helper Functions ====================

def round_decimal(value, decimal_places=2):
    """
    Round a float to a specified number of decimal places using Decimal for precision.
    """
    quantize_str = '1.' + '0' * decimal_places
    return float(Decimal(value).quantize(Decimal(quantize_str), rounding=ROUND_DOWN))

def submit_order_with_retry(order_params, max_retries=3, delay=2):
    """
    Submit an order with a retry mechanism in case of transient failures.
    """
    for attempt in range(max_retries):
        try:
            return alpaca_utils.api.submit_order(**order_params)
        except Exception as e:
            logging.error(f"Order submission failed on attempt {attempt + 1}: {e}")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
    logging.error("All retry attempts failed. Order not placed.")
    return None

# ==================== Trading Decision Function ====================

def make_trading_decision(data):
    global position, buy_order_id, last_trade_time, trades

    # Ensure we have enough data
    min_required = max(
        INDICATORS_CONFIG["RSI"]["window"],
        INDICATORS_CONFIG["MACD"]["slow"],
    )

    if len(data) < min_required:
        logging.debug("Not enough data to make a trading decision.")
        return

    # ----- Position Status Checking -----
    try:
        # Fetch current positions from Alpaca
        current_positions = alpaca_utils.api.list_positions()
        # Check if currently holding a position in SYMBOL
        holding_position = any(pos.symbol == SYMBOL for pos in current_positions)
        position = 'long' if holding_position else None

    except Exception as e:
        logging.error(f"Error checking current positions: {e}")
        return  # Exit the function if unable to verify positions

    # ----- Cooldown Check -----
    current_time = datetime.utcnow()
    if last_trade_time and (current_time - last_trade_time) < COOLDOWN_PERIOD:
        logging.debug("Cooldown period active. Skipping trade decision.")
        return

    # ----- Indicator Extraction -----
    latest = data.iloc[-1]

    rsi_window = INDICATORS_CONFIG["RSI"]["window"]
    macd_fast = INDICATORS_CONFIG["MACD"]["fast"]
    macd_slow = INDICATORS_CONFIG["MACD"]["slow"]
    macd_signal = INDICATORS_CONFIG["MACD"]["signal"]

    current_rsi = latest[f"RSI_{rsi_window}"]
    current_macd = latest["MACD"]
    current_macd_signal = latest["MACD_Signal"]

    previous_rsi = data.iloc[-2][f"RSI_{rsi_window}"] if len(data) > 1 else current_rsi
    previous_macd = data.iloc[-2]["MACD"] if len(data) > 1 else current_macd
    previous_macd_signal = data.iloc[-2]["MACD_Signal"] if len(data) > 1 else current_macd_signal

    # ----- Buy Signal Determination -----
    buy_signal = False

    # RSI crossing above 30
    rsi_oversold = previous_rsi < 30 and current_rsi >= 30

    # MACD bullish crossover
    macd_bullish_cross = previous_macd <= previous_macd_signal and current_macd > current_macd_signal

    if rsi_oversold and macd_bullish_cross and position is None:
        buy_signal = True
        logging.info(f"Buy signal generated at ${latest['close']:.2f} on {latest['Datetime']}")

    # ----- Sell Signal Determination -----
    sell_signal = False

    # RSI crossing below 70
    rsi_overbought = previous_rsi > 70 and current_rsi <= 70

    # MACD bearish crossover
    macd_bearish_cross = previous_macd >= previous_macd_signal and current_macd < current_macd_signal

    if position == 'long' and (rsi_overbought or macd_bearish_cross):
        sell_signal = True
        logging.info(f"Sell signal generated at ${latest['close']:.2f} on {latest['Datetime']}")

    # ----- Execute Buy Order -----
    if buy_signal and position is None:
        try:
            # Fetch account information
            account = alpaca_utils.api.get_account()
            buying_power = float(account.buying_power)

            # Allocate 10% of buying power
            position_size = buying_power * 0.10

            # Calculate the number of shares to buy
            qty = int(position_size / latest['close'])
            qty = max(qty, 1)  # Ensure at least 1 share

            # Prepare order parameters
            order_params = {
                'symbol': SYMBOL,
                'qty': qty,
                'side': 'buy',
                'type': 'market',
                'time_in_force': 'gtc',
            }

            # Place the market order
            order = submit_order_with_retry(order_params)
            if order:
                position = 'long'
                buy_order_id = order.id  # Store the order ID for tracking
                last_trade_time = current_time  # Update the last trade time

                # Log the buy event
                trades.append({
                    'type': 'buy',
                    'price': latest['close'],
                    'datetime': latest['Datetime']
                })
                logging.info(f"Executed Buy Order: {order}")

        except Exception as e:
            logging.error(f"Error executing buy order: {e}")

    # ----- Execute Sell Order -----
    if sell_signal and position == 'long':
        try:
            # Fetch current positions to get the quantity
            positions = alpaca_utils.api.list_positions()
            for pos in positions:
                if pos.symbol == SYMBOL:
                    qty = pos.qty

                    # Prepare sell order parameters
                    order_params = {
                        'symbol': SYMBOL,
                        'qty': qty,
                        'side': 'sell',
                        'type': 'market',
                        'time_in_force': 'gtc',
                    }

                    # Place the market sell order
                    order = submit_order_with_retry(order_params)
                    if order:
                        position = None
                        last_trade_time = current_time  # Update the last trade time

                        # Log the sell event
                        trades.append({
                            'type': 'sell',
                            'price': latest['close'],
                            'datetime': latest['Datetime']
                        })
                        logging.info(f"Executed Sell Order: {order}")

        except Exception as e:
            logging.error(f"Error executing sell order: {e}")


    # ----- Execute Sell Order -----
    if sell_signal and position == 'long':
        try:
            # Fetch current positions to get the quantity
            positions = alpaca_utils.api.list_positions()
            for pos in positions:
                if pos.symbol == SYMBOL:
                    qty = pos.qty

                    # Prepare sell order parameters
                    order_params = {
                        'symbol': SYMBOL,
                        'qty': qty,
                        'side': 'sell',
                        'type': 'market',
                        'time_in_force': 'gtc',
                    }

                    # Place the market sell order
                    order = submit_order_with_retry(order_params)
                    if order:
                        position = None
                        last_trade_time = current_time  # Update the last trade time

                        # Log the sell event
                        trades.append({
                            'type': 'sell',
                            'price': latest['close'],
                            'datetime': latest['Datetime']
                        })
                        logging.info(f"Executed Sell Order: {order}")

        except Exception as e:
            logging.error(f"Error executing sell order: {e}")


    # ----- Monitor Order Status -----
    if buy_order_id and position == 'long':
        try:
            buy_order = alpaca_utils.api.get_order(buy_order_id)
            if buy_order.status == 'filled':
                logging.info(f"Buy order {buy_order_id} filled at ${buy_order.filled_avg_price}")
                buy_order_id = None  # Reset after handling the filled order
            elif buy_order.status in ['canceled', 'rejected']:
                logging.info(f"Buy order {buy_order_id} {buy_order.status}. Resetting position.")
                position = None
                buy_order_id = None
        except Exception as e:
            logging.error(f"Error checking buy order status: {e}")

# ==================== Plot Update Function ====================

def update_plot(frame):
    global data_with_indicators, position, trades

    # Fetch WebSocket data
    price_data = client.get_price_data()
    if SYMBOL in price_data:
        # Safely extract price and timestamp
        latest_price = price_data[SYMBOL].get("price", None)
        latest_timestamp_str = price_data[SYMBOL].get("timestamp", None)

        if latest_price is None or latest_timestamp_str is None:
            logging.warning("Incomplete data received from WebSocket. Skipping update.")
            return

        # Parse the timestamp using dateutil and convert to timezone-naive UTC
        try:
            parsed_timestamp = isoparse(latest_timestamp_str)
            latest_timestamp = parsed_timestamp.astimezone(timezone.utc).replace(tzinfo=None)
        except Exception as e:
            logging.error(f"Error parsing timestamp: {e}")
            latest_timestamp = datetime.utcnow()

        # Create a new DataFrame row with the latest data
        new_row = pd.DataFrame({
            'Datetime': [latest_timestamp],
            'open': [latest_price],
            'high': [latest_price],
            'low': [latest_price],
            'close': [latest_price],
            'volume': [0],
        })

        # Append the new row to the existing DataFrame
        data_with_indicators = pd.concat([data_with_indicators, new_row], ignore_index=True)
        logging.debug(f"Appended new row: {new_row.iloc[0].to_dict()}")

        # Recalculate indicators
        data_with_indicators = alpaca_utils.calculate_indicators(
            data_with_indicators, INDICATORS_CONFIG
        )

        if data_with_indicators is None:
            logging.error("Indicator calculation failed. Data_with_indicators is None.")
            return

        # Make trading decisions
        make_trading_decision(data_with_indicators)

        # Sort data by Datetime to ensure correct plotting
        data_with_indicators = data_with_indicators.sort_values(by="Datetime").reset_index(drop=True)

        # Limit the data to recent N data points or last T minutes
        MAX_DATA_POINTS = 60  # e.g., last 60 minutes
        if len(data_with_indicators) > MAX_DATA_POINTS:
            data_with_indicators = data_with_indicators.tail(MAX_DATA_POINTS)

        # Sort trades to only keep recent ones within the plot window
        if trades:
            # Remove trades that are outside the current plotting window
            plot_start_time = data_with_indicators['Datetime'].iloc[0]
            trades = [trade for trade in trades if trade['datetime'] >= plot_start_time]

        # Plotting
        ax.clear()

        # Plot the price data
        ax.plot(data_with_indicators['Datetime'], data_with_indicators['close'], label=f"{SYMBOL} Price")

        # Plot the SMA indicator
        sma_column = f"SMA_{INDICATORS_CONFIG['SMA']['window']}"
        if sma_column in data_with_indicators.columns:
            ax.plot(
                data_with_indicators['Datetime'],
                data_with_indicators[sma_column],
                label=sma_column
            )

        # Plot the EMA indicator
        ema_column = f"EMA_{INDICATORS_CONFIG['EMA']['window']}"
        if ema_column in data_with_indicators.columns:
            ax.plot(
                data_with_indicators['Datetime'],
                data_with_indicators[ema_column],
                label=ema_column
            )

        # Plot Bollinger Bands
        bollinger_high = f"Bollinger_High_{INDICATORS_CONFIG['Bollinger']['window']}"
        bollinger_low = f"Bollinger_Low_{INDICATORS_CONFIG['Bollinger']['window']}"
        if bollinger_high in data_with_indicators.columns and bollinger_low in data_with_indicators.columns:
            ax.plot(
                data_with_indicators['Datetime'],
                data_with_indicators[bollinger_high],
                label=bollinger_high,
                linestyle='--',
                color='grey'
            )
            ax.plot(
                data_with_indicators['Datetime'],
                data_with_indicators[bollinger_low],
                label=bollinger_low,
                linestyle='--',
                color='grey'
            )

        # Plot buy and sell markers
        for trade in trades:
            if trade['type'] == 'buy':
                ax.plot(trade['datetime'], trade['price'], marker='^', markersize=10, color='green', label='Buy')
            elif trade['type'] == 'sell':
                ax.plot(trade['datetime'], trade['price'], marker='v', markersize=10, color='red', label='Sell')

        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.set_title("Live Price Data with Indicators")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")

        # Adjust x-axis date formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # Automatically adjust axis limits
        ax.relim()
        ax.autoscale_view()

        # Ensure layout fits
        plt.tight_layout()

# ==================== Main Function ====================

def main():
    global data_with_indicators

    # Fetch historical data
    logging.info(f"Fetching historical data for {SYMBOL}")
    data = alpaca_utils.fetch_historical_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        limit=LIMIT
    )

    if data is not None and not data.empty:
        logging.info(f"Successfully fetched data for {SYMBOL}: {len(data)} rows.")
        logging.info("Fetched historical data:")
        logging.info(data.tail())

        # Calculate indicators
        data_with_indicators = alpaca_utils.calculate_indicators(data, INDICATORS_CONFIG)
        data_with_indicators['Datetime'] = pd.to_datetime(data_with_indicators['Datetime'])
        data_with_indicators = data_with_indicators.sort_values(by="Datetime").reset_index(drop=True)
        data_with_indicators['Datetime'] = data_with_indicators['Datetime'].dt.tz_localize(None)
    else:
        logging.error("Failed to fetch historical data or calculate indicators.")
        # Initialize an empty DataFrame with required columns for live updates
        data_with_indicators = pd.DataFrame(
            columns=['Datetime', 'open', 'high', 'low', 'close', 'volume']
        )

    # Start the live plot
    ani = FuncAnimation(fig, update_plot, interval=500)
    plt.show()

if __name__ == "__main__":
    main()
