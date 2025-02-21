import pandas as pd
import asyncio
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor
from config import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_DATA_URL,
    TIMEFRAMES,
    HISTORICAL_BARS,
    VERBOSE_DATA,
    SYMBOLS
)

class DataFetcher:
    def __init__(self):
            
        self.client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
            url_override=ALPACA_DATA_URL
        )
        

    async def _fetch_symbol_data(self, symbol: str, start: datetime, end: datetime) -> dict:
        """Fetch and process data for a single symbol."""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start,
                end=end,
                adjustment='all',
                feed='sip'
            )
            
            # Run API request in thread pool to prevent blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as pool:
                bars = await loop.run_in_executor(
                    pool, 
                    self.client.get_stock_bars,
                    request
                )
            
            if not bars:
                if VERBOSE_DATA:
                    print(f"No data received for {symbol}")
                return {}
            
            # Convert bars to DataFrame
            try:
                symbol_bars = bars[symbol]
                df = pd.DataFrame([{
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'trade_count': bar.trade_count,
                    'vwap': bar.vwap
                } for bar in symbol_bars])
            except (KeyError, AttributeError) as e:
                if VERBOSE_DATA:
                    print(f"No data available for {symbol}: {str(e)}")
                return {}
            
            if df.empty:
                if VERBOSE_DATA:
                    print(f"No data available for {symbol}")
                return {}
            
            # Set timestamp as index and sort
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            
            # Convert to different timeframes
            timeframe_data = {}
            
            # Store 5-minute bars
            timeframe_data['short'] = df
            
            # Create 15-minute bars
            timeframe_data['medium'] = self._aggregate_bars(df, '15min')
            
            # Create 60-minute bars
            timeframe_data['long'] = self._aggregate_bars(df, '60min')
            
            
            return timeframe_data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return {}

    def _aggregate_bars(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Aggregate bars to a larger timeframe."""
        # Resample rules
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'trade_count': 'sum',
            'vwap': 'mean'
        }
        
        return df.resample(timeframe).agg(agg_dict)

    async def update_data(self, symbols: list = None) -> dict:
        """Fetch and process market data for all timeframes concurrently."""
        if symbols is None:
            symbols = SYMBOLS
            
        # Calculate time range
        # Add 3 hours to convert PST to EST
        now = datetime.now()
        end = now.replace(hour=(now.hour + 3) % 24)
        calendar_days = 62  # ~2 months to ensure enough data for 200 hourly bars
        start = end - timedelta(days=calendar_days)
        

        
        # Fetch data for all symbols concurrently
        tasks = [self._fetch_symbol_data(symbol, start, end) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        # Combine results
        return {symbol: data for symbol, data in zip(symbols, results) if data}
