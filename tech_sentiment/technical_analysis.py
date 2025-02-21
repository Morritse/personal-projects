import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import talib
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide
from dotenv import load_dotenv

class TechnicalAnalyzer:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('ALPACA_API_KEY')
        api_secret = os.getenv('ALPACA_SECRET_KEY')
        self.client = StockHistoricalDataClient(api_key, api_secret)
        
    def get_historical_data(self, symbol, lookback_days=100):
        """Get historical data from Alpaca"""
        try:
            # Get today's date in ET
            et_tz = datetime.now().astimezone().tzinfo
            today = datetime.now(et_tz).date()
            
            # Set end to today's market close (4 PM ET)
            end = datetime.combine(today, datetime.strptime('16:00', '%H:%M').time())
            end = end.replace(tzinfo=et_tz)
            
            # Set start to lookback days before
            start = end - timedelta(days=lookback_days)
            
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                adjustment='all'
            )
            
            bars = self.client.get_stock_bars(request)
            df = bars.df
            return df if not df.empty else None
            
        except Exception as e:
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        # Convert multi-index to single index
        df = df.reset_index(level=1, drop=True)
        
        # Basic indicators
        df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD
        macd, macd_signal, _ = talib.MACD(df['close'])
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'])
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        
        return df
    
    def analyze_stock(self, symbol, side=None):
        """Analyze a stock and return trading signals"""
        df = self.get_historical_data(symbol)
        if df is None:
            return None
            
        df = self.calculate_indicators(df)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate 24h changes
        price_24h_change = ((latest['close'] / prev['close']) - 1) * 100
        
        signals = {
            'symbol': symbol,
            'price': latest['close'],
            'signals': [],
            'raw_score': 0,
            'score': 0,
            'momentum': price_24h_change,
            'exit_signals': []  # For tracking exit reasons
        }
        
        # Price vs moving averages
        price = latest['close']
        sma20 = latest['SMA_20']
        sma50 = latest['SMA_50']
        
        if price > sma20 and price > sma50:
            if sma20 > sma50:
                signals['score'] += 30
            else:
                signals['score'] += 10
        else:
            if price < sma20 and price < sma50:
                signals['score'] -= 30
                signals['signals'].append("below both MAs")
            else:
                signals['score'] -= 10
        
        # RSI Analysis
        if latest['RSI'] < 30:
            if side != OrderSide.SELL:
                signals['score'] += 30
            else:
                signals['score'] -= 15
        elif latest['RSI'] > 70:
            if side != OrderSide.BUY:
                signals['score'] -= 30
            else:
                signals['score'] += 15
        
        # MACD Analysis
        macd = latest['MACD']
        signal = latest['MACD_Signal']
        macd_diff = macd - signal
        
        if abs(macd_diff) < 0.1:
            signals['score'] -= 10
        elif macd > signal:
            if macd_diff > 0.5:
                signals['score'] += 30
            else:
                signals['score'] += 10
        else:
            if macd_diff < -0.5:
                signals['score'] -= 30
                signals['signals'].append("Strong bearish MACD")
            else:
                signals['score'] -= 10
        
        # Bollinger Bands Analysis
        if latest['close'] < latest['BB_Lower']:
            if side != OrderSide.SELL:
                signals['score'] += 25
            else:
                signals['score'] -= 15
        elif latest['close'] > latest['BB_Upper']:
            if side != OrderSide.BUY:
                signals['score'] -= 25
            else:
                signals['score'] += 15
        
        # 24h momentum analysis
        if signals['momentum'] > 2:
            signals['score'] += 30
        elif signals['momentum'] > 0:
            signals['score'] += 15
        elif signals['momentum'] > -2:
            signals['score'] -= 15
        else:
            signals['score'] -= 30
            
        # Normalize score from -100,100 to 0,1 range
        signals['raw_score'] = signals['score']
        signals['score'] = (signals['score'] + 100) / 200
        return signals

def analyze_hype_stocks(hype_stocks):
    """Analyze technical indicators for hype stocks"""
    analyzer = TechnicalAnalyzer()
    results = []
    
    for stock in hype_stocks:
        technical_data = analyzer.analyze_stock(stock['ticker'])
        if technical_data:
            combined_score = (stock['average_sentiment'] * 50) + (technical_data['score'] * 0.5)
            results.append({
                'ticker': stock['ticker'],
                'sentiment_score': stock['average_sentiment'],
                'technical_score': technical_data['score'],
                'combined_score': combined_score,
                'price': technical_data['price'],
                'technical_signals': technical_data['signals'],
                'recent_news': stock['recent_news']
            })
    
    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results
