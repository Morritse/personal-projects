"""
Market data analyzer that provides a structured format for passing trading data to Claude.
This allows for consistent analysis of market conditions, technical indicators, and patterns.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@dataclass
class MarketData:
    """Container for market data and technical indicators"""
    # Metadata
    timestamp: str  # Current candle timestamp
    symbol: str    # Trading symbol (e.g., AAPL, MSFT)
    timeframe: str # e.g., "1h", "4h", "1d"
    
    # Price data
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Trend indicators
    sma_20: float           # 20-period Simple Moving Average
    sma_50: float           # 50-period Simple Moving Average
    sma_200: float          # 200-period Simple Moving Average
    atr_14: float           # 14-period Average True Range
    
    # Momentum indicators
    rsi_14: float           # 14-period Relative Strength Index
    macd: Dict[str, float]  # MACD(12,26,9) - Contains 'macd', 'signal', 'hist'
    
    # Volume indicators
    vwap: float            # Volume Weighted Average Price
    obv: float             # On Balance Volume
    
    # Volatility
    bb_bands: Dict[str, float]  # 20-period Bollinger Bands with 2 std dev
    
    # Support/Resistance
    pivot_points: Dict[str, float]  # Classical pivot points (P, S1-S3, R1-R3)
    
    # Market data
    market_cap: float              # Company market capitalization
    sector: str                    # Company sector
    beta: float                    # Beta value relative to market
    
    # Fundamental metrics (updated quarterly)
    pe_ratio: Optional[float] = None      # Price to Earnings ratio
    pb_ratio: Optional[float] = None      # Price to Book ratio
    debt_to_equity: Optional[float] = None
    
    # Recent events
    earnings_date: Optional[str] = None   # Next earnings date
    dividend_date: Optional[str] = None   # Next dividend date
    stock_splits: Optional[List[Dict[str, str]]] = None  # Recent stock splits
    
    def to_json(self) -> str:
        """Convert to JSON string for easy transmission"""
        def convert_numpy(obj):
            if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            return obj
        
        data = {k: convert_numpy(v) for k, v in self.__dict__.items()}
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'MarketData':
        """Create MarketData instance from JSON string"""
        data = json.loads(json_str)
        return cls(**data)

# List of symbols we'll track
SYMBOLS = [
    # Large cap tech
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    
    # Semiconductor
    "NVDA",  # NVIDIA
    "AMD",   # Advanced Micro Devices
    
    # ETFs
    "SPY",   # S&P 500
    "QQQ",   # Nasdaq 100
    "VIX",   # Volatility Index
]

# Timeframes to analyze
TIMEFRAMES = [
    "1h",    # Intraday momentum
    "4h",    # Intraday trend
    "1d"     # Swing trading
]

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators for the given price data"""
    # Moving averages
    df['sma_20'] = df['Close'].rolling(window=20).mean()
    df['sma_50'] = df['Close'].rolling(window=50).mean()
    df['sma_200'] = df['Close'].rolling(window=200).mean()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr_14'] = true_range.rolling(14).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = macd - signal
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (std * 2)
    df['bb_lower'] = df['bb_middle'] - (std * 2)
    
    return df

def fetch_market_data(symbol: str) -> str:
    """Fetch real market data from yfinance"""
    # Get stock info
    stock = yf.Ticker(symbol)
    info = stock.info
    
    # Get historical data (1 year)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    df = stock.history(start=start_date, end=end_date, interval="1d")
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Get the latest row
    latest = df.iloc[-1]
    
    # Create MarketData object
    data = MarketData(
        timestamp=latest.name.strftime("%Y-%m-%dT%H:%M:%SZ"),
        symbol=symbol,
        timeframe="1d",
        open=latest['Open'],
        high=latest['High'],
        low=latest['Low'],
        close=latest['Close'],
        volume=latest['Volume'],
        
        # Technical indicators
        sma_20=latest['sma_20'],
        sma_50=latest['sma_50'],
        sma_200=latest['sma_200'],
        atr_14=latest['atr_14'],
        rsi_14=latest['rsi_14'],
        macd={
            "macd": latest['macd'],
            "signal": latest['macd_signal'],
            "hist": latest['macd_hist']
        },
        
        # Volume indicators
        vwap=df['Close'].mean(),  # Simple approximation
        obv=df['Volume'].sum(),   # Simple approximation
        
        # Volatility
        bb_bands={
            "upper": latest['bb_upper'],
            "middle": latest['bb_middle'],
            "lower": latest['bb_lower']
        },
        
        # Support/Resistance (using recent highs/lows)
        pivot_points={
            "P": latest['Close'],
            "R1": df['High'].tail(20).max(),
            "R2": df['High'].tail(50).max(),
            "R3": df['High'].tail(200).max(),
            "S1": df['Low'].tail(20).min(),
            "S2": df['Low'].tail(50).min(),
            "S3": df['Low'].tail(200).min()
        },
        
        # Market data
        market_cap=info.get('marketCap', 0),
        sector=info.get('sector', 'Unknown'),
        beta=info.get('beta', 0),
        
        # Fundamental metrics
        pe_ratio=info.get('trailingPE', None),
        pb_ratio=info.get('priceToBook', None),
        debt_to_equity=info.get('debtToEquity', None),
        
        # Events
        earnings_date=info.get('earningsDate', [None])[0].strftime("%Y-%m-%d") if info.get('earningsDate') else None,
        dividend_date=info.get('dividendDate', None),
        stock_splits=[{"date": str(date), "ratio": str(ratio)} 
                     for date, ratio in stock.splits.items()]
    )
    return data.to_json()

def main():
    """Fetch and analyze real market data"""
    try:
        # Fetch real data for a symbol
        symbol = "AAPL"  # Can be changed to any symbol in SYMBOLS list
        print(f"\nFetching real market data for {symbol}...")
        
        json_data = fetch_market_data(symbol)
        print("\nSample market data JSON:")
        print(json_data)
        
        # Parse JSON back into MarketData object
        market_data = MarketData.from_json(json_data)
        
        print("\nTechnical Analysis:")
        print(f"Symbol: {market_data.symbol} ({market_data.timeframe} timeframe)")
        print(f"Price: ${market_data.close:.2f}")
        print(f"Trend Status: SMA20 @ ${market_data.sma_20:.2f}, SMA50 @ ${market_data.sma_50:.2f}, SMA200 @ ${market_data.sma_200:.2f}")
        print(f"RSI: {market_data.rsi_14:.1f}")
        print(f"MACD: {market_data.macd['hist']:.3f}")
        
        print("\nSupport/Resistance:")
        print(f"Next resistance levels: ${market_data.pivot_points['R1']:.2f}, ${market_data.pivot_points['R2']:.2f}")
        print(f"Next support levels: ${market_data.pivot_points['S1']:.2f}, ${market_data.pivot_points['S2']:.2f}")
        
        print("\nFundamental Data:")
        print(f"Market Cap: ${market_data.market_cap / 1e12:.2f}T")
        print(f"P/E Ratio: {market_data.pe_ratio:.1f}")
        print(f"Next Earnings: {market_data.earnings_date}")
        
    except Exception as e:
        print(f"Error analyzing market data: {str(e)}")

if __name__ == "__main__":
    main()
