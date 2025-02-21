import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json

class StocktwitsScanner:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_trending_stocks(self, limit=20):
        """Get trending stocks from Stocktwits"""
        try:
            url = 'https://api.stocktwits.com/api/2/trending/symbols.json'
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                return pd.DataFrame()
            
            try:
                data = response.json()
                symbols = data.get('symbols', [])
                
                # Convert to DataFrame
                df = pd.DataFrame(symbols)
                if df.empty:
                    return df
                
                # Rename columns
                df = df.rename(columns={
                    'symbol': 'ticker',
                    'watchlist_count': 'mentions'
                })
                
                # Filter to only stocks
                df = df[df['instrument_class'] == 'Stock']
                
                # Get sentiment from messages
                def get_message_sentiment(ticker):
                    try:
                        messages_url = f'https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json'
                        response = requests.get(messages_url, headers=self.headers)
                        if response.status_code == 200:
                            data = response.json()
                            messages = data.get('messages', [])
                            if messages:
                                bullish = 0
                                bearish = 0
                                total = 0
                                
                                for message in messages[:100]:
                                    if 'entities' in message:
                                        sentiment = message['entities'].get('sentiment', {})
                                        if sentiment and 'basic' in sentiment:
                                            total += 1
                                            if sentiment['basic'] == 'Bullish':
                                                bullish += 1
                                            elif sentiment['basic'] == 'Bearish':
                                                bearish += 1
                                
                                if total > 0:
                                    return bullish / total
                        return 0.5
                    except:
                        return 0.5
                
                df['bullish_ratio'] = df['ticker'].apply(get_message_sentiment)
                
                # Filter for minimum watchers and bullish ratio
                df = df[df['mentions'] >= 1000]
                df = df[df['bullish_ratio'] > 0.6]
                
                if df.empty:
                    return df
                
                # Calculate score as log10(watchers) * bullish_ratio
                df['log_mentions'] = np.log10(df['mentions'])
                df['score'] = df['log_mentions'] * df['bullish_ratio']
                
                # Sort by score
                df = df.sort_values('score', ascending=False)
                df['rank'] = range(1, len(df) + 1)
                
                # Add source and timestamp
                df['source'] = 'stocktwits'
                df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Get top N stocks
                return df.head(limit)
                
            except:
                return pd.DataFrame()
                
        except:
            return pd.DataFrame()
