# get_data.py

import os
import time
import pandas as pd
from web3 import Web3
from tqdm import tqdm
import logging
import concurrent.futures
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataCollector:
    def __init__(self):
        # Infura URL with API key as plain text
        self.INFURA_URL = "https://mainnet.infura.io/v3/3a1f55e4f5224900965a893eb212875e"
        self.w3 = Web3(Web3.HTTPProvider(self.INFURA_URL))

        if not self.w3.isConnected():
            logger.error("Unable to connect to Infura endpoint.")
            raise ConnectionError("Failed to connect to Infura.")

        logger.info("Connected to Infura.")

        # Token addresses (USDC and USDT on Ethereum)
        self.TOKENS = {
            'USDC': '0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48',
            'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7'
        }

        # Standard ERC20 transfer event signature
        self.TRANSFER_EVENT = self.w3.keccak(text='Transfer(address,address,uint256)').hex()

        # Initialize storage
        self.whale_addresses = set()
        self.transaction_history = pd.DataFrame()

        # Lock for thread-safe operations
        self.lock = Lock()

    def get_large_transfers(self, token, from_block, to_block, threshold_usd=100000):
        """Get large transfers for a given token within a block range."""
        transfer_filter = {
            'fromBlock': from_block,
            'toBlock': to_block,
            'address': self.TOKENS[token],
            'topics': [self.TRANSFER_EVENT]
        }

        try:
            events = self.w3.eth.get_logs(transfer_filter)
            transfer_data = []

            logger.debug(f"Processing {len(events)} events for {token} from blocks {from_block} to {to_block}.")

            for event in events:
                try:
                    from_addr = '0x' + event['topics'][1].hex()[-40:]
                    to_addr = '0x' + event['topics'][2].hex()[-40:]
                    value = int(event['data'], 16)

                    # Assume 6 decimal places for both USDC and USDT
                    value_usd = value / (10**6)

                    if value_usd >= threshold_usd:
                        transfer_data.append({
                            'token': token,
                            'from': from_addr,
                            'to': to_addr,
                            'amount_usd': value_usd,
                            'block_number': event['blockNumber'],
                            'timestamp': self.w3.eth.get_block(event['blockNumber'])['timestamp']
                        })

                except Exception as e:
                    logger.error(f"Error processing event: {str(e)}")
                    continue

            if not transfer_data:
                logger.debug(f"No transfers above {threshold_usd} USD found in blocks {from_block} to {to_block}.")
                return pd.DataFrame()

            logger.debug(f"Found {len(transfer_data)} transfers above {threshold_usd} USD in blocks {from_block} to {to_block}.")
            return pd.DataFrame(transfer_data)

        except Exception as e:
            logger.error(f"Error fetching large transfers for {token}: {str(e)}")
            raise e  # Re-raise exception to handle it in the caller

    def safe_get_large_transfers(self, token, from_block, to_block, threshold_usd=100000,
                                 max_results=50, min_block_range=5, current_depth=0, max_depth=15):
        """
        Fetch large transfers with dynamic block range splitting to handle Infura's result limits.
        Includes a maximum recursion depth to prevent infinite splitting.
        """
        try:
            df = self.get_large_transfers(token, from_block, to_block, threshold_usd)
            logger.info(f"Fetched {len(df)} transfers for {token} from blocks {from_block} to {to_block}.")

            # If fetched transfers exceed max_results, split the block range
            if len(df) > max_results:
                if (to_block - from_block) <= min_block_range:
                    logger.warning(f"Reached minimum block range with {len(df)} results for {token}.")
                    return df
                if current_depth >= max_depth:
                    logger.error(f"Maximum recursion depth reached for {token}. Returning current data.")
                    return df
                mid_block = from_block + (to_block - from_block) // 2
                logger.info(f"Splitting block range [{from_block}, {to_block}] into "
                            f"[{from_block}, {mid_block}] and [{mid_block + 1}, {to_block}] for {token}.")
                df1 = self.safe_get_large_transfers(token, from_block, mid_block, threshold_usd,
                                                   max_results, min_block_range, current_depth + 1, max_depth)
                df2 = self.safe_get_large_transfers(token, mid_block + 1, to_block, threshold_usd,
                                                   max_results, min_block_range, current_depth + 1, max_depth)
                return pd.concat([df1, df2], ignore_index=True)
            return df

        except Exception as e:
            # Handle specific Infura errors by checking exception messages
            error_msg = str(e).lower()
            if "query returned more than" in error_msg or "result limit" in error_msg:
                if (to_block - from_block) <= min_block_range:
                    logger.warning(f"Reached minimum block range with error: {str(e)}")
                    return pd.DataFrame()
                if current_depth >= max_depth:
                    logger.error(f"Maximum recursion depth reached for {token} due to error: {str(e)}")
                    return pd.DataFrame()
                mid_block = from_block + (to_block - from_block) // 2
                logger.info(f"Splitting block range [{from_block}, {to_block}] due to error into "
                            f"[{from_block}, {mid_block}] and [{mid_block + 1}, {to_block}] for {token}.")
                df1 = self.safe_get_large_transfers(token, from_block, mid_block, threshold_usd,
                                                   max_results, min_block_range, current_depth + 1, max_depth)
                df2 = self.safe_get_large_transfers(token, mid_block + 1, to_block, threshold_usd,
                                                   max_results, min_block_range, current_depth + 1, max_depth)
                return pd.concat([df1, df2], ignore_index=True)
            else:
                logger.error(f"Unhandled exception: {str(e)}")
                return pd.DataFrame()

    def get_historical_transfers(self, token, days=1, threshold_usd=150000):
        """Fetch historical large transfers for a given token over a number of days."""
        end_time = int(time.time())
        start_time = end_time - days * 86400  # Convert days to seconds

        try:
            latest_block = w3.eth.block_number
            latest_block_timestamp = w3.eth.get_block(latest_block)['timestamp']

            # Estimate the starting block by subtracting average blocks per day
            avg_blocks_per_day = 6500
            start_block = max(latest_block - days * avg_blocks_per_day, 0)

            logger.info(f"Fetching historical data for {token} from block {start_block} to block {latest_block}...")

            transfer_data = []

            # Define initial chunk size (number of blocks per request)
            initial_chunk_size = 100  # Further reduced
            min_block_range = 5        # Allow finer splitting
            max_depth = 10             # Reduced recursion depth

            total_blocks = latest_block - start_block + 1

            # Generate all block ranges
            block_ranges = []
            current_from = start_block
            while current_from <= latest_block:
                current_to = min(current_from + initial_chunk_size - 1, latest_block)
                block_ranges.append((current_from, current_to))
                current_from = current_to + 1

            transfer_data = []

            with tqdm(total=total_blocks, desc=f"Fetching {token} transfers") as pbar:
                # Define the maximum number of threads
                max_workers = 6  # Adjust based on Infura's rate limits

                def fetch_and_update(from_block, to_block):
                    df = safe_get_large_transfers(token, from_block, to_block, threshold_usd)
                    if not df.empty:
                        with self.lock:
                            transfer_data.append(df)
                    pbar.update(to_block - from_block + 1)

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(fetch_and_update, fr, to) for fr, to in block_ranges]
                    concurrent.futures.wait(futures)

            if transfer_data:
                historical_transfers = pd.concat(transfer_data, ignore_index=True)
                logger.info(f"Collected {len(historical_transfers)} large transfers for {token}.")
                return historical_transfers
            else:
                logger.warning(f"No large transfers found for {token} in the specified period.")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching historical transfers for {token}: {str(e)}")
            return pd.DataFrame()

    def identify_whales(self, threshold_usd=150000, days=1):
        """Identify whale addresses based on transfer volume threshold."""
        whales = set()
        for token in self.TOKENS:
            df = self.get_historical_transfers(token, days=days, threshold_usd=threshold_usd)
            if not df.empty:
                whales.update(df['from'].tolist())
                whales.update(df['to'].tolist())
        self.whale_addresses = whales
        logger.info(f"Identified {len(self.whale_addresses)} whale addresses.")

    def collect_real_time_data(self):
        """Collects new transactions and updates transaction history for real-time analysis."""
        all_new_transfers = []
        for token in self.TOKENS:
            try:
                # Fetch the latest 100 blocks
                latest_block = self.w3.eth.block_number
                from_block = max(latest_block - 100, 0)
                to_block = latest_block

                new_transfers = self.safe_get_large_transfers(token, from_block, to_block, threshold_usd=150000)
                if not new_transfers.empty:
                    all_new_transfers.append(new_transfers)
                logger.info(f"Updated {token} transfers. Found {len(new_transfers)} new transfers.")
            except Exception as e:
                logger.error(f"Error tracking {token}: {str(e)}")

        if all_new_transfers:
            self.transaction_history = pd.concat(
                [self.transaction_history] + all_new_transfers,
                ignore_index=True
            )

            # Clean old data (keep only recent 24 hours)
            current_time = int(time.time())
            self.transaction_history = self.transaction_history[
                self.transaction_history['timestamp'] > current_time - 86400
            ]

            # Update whale addresses based on new data
            self.identify_whales(threshold_usd=150000, days=1)  # Update whales based on the last day

        return self.transaction_history

# Instantiate Web3 outside the class for use in get_historical_transfers
w3 = Web3(Web3.HTTPProvider("https://mainnet.infura.io/v3/3a1f55e4f5224900965a893eb212875e"))

def safe_get_large_transfers(token, from_block, to_block, threshold_usd):
    """Wrapper function to handle exceptions during transfer fetching."""
    collector = CryptoDataCollector()
    return collector.safe_get_large_transfers(token, from_block, to_block, threshold_usd)
