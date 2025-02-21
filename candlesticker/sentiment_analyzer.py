"""
Fetch and analyze news sentiment for trading targets.
"""

import os
import json
import re
from datetime import datetime, timedelta
import requests

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
if not NEWS_API_KEY:
    raise ValueError("NEWS_API_KEY environment variable is required")

# Known company mappings
company_names = {
    'AAPL': 'Apple',
    'MSFT': 'Microsoft',
    'GOOGL': 'Google OR Alphabet',
    'NVDA': 'Nvidia',
    'AMD': 'AMD OR "Advanced Micro Devices"',
    'META': 'Meta OR Facebook',
    'AMZN': 'Amazon',
    'NFLX': 'Netflix',
    'TSLA': 'Tesla',
    'INTC': 'Intel'
}

def get_news_sentiment(symbol: str, days: int = 3) -> dict:
    """Get news sentiment for a symbol"""
    from_date = (datetime.now() - timedelta(days=days)).isoformat()
    
    try:
        # For unknown symbols, try to find company name from news
        if symbol not in company_names:
            # Search for any news mentioning the symbol
            search_response = requests.get(
                'https://newsapi.org/v2/everything',
                params={
                    'q': f'"{symbol}" AND (stock OR shares)',
                    'language': 'en',
                    'sortBy': 'relevancy',
                    'pageSize': 1
                },
                headers={
                    'X-Api-Key': NEWS_API_KEY,
                    'User-Agent': 'Mozilla/5.0'
                }
            )
            search_response.raise_for_status()
            
            # Try to extract company name from first article
            articles = search_response.json()['articles']
            if articles:
                # Use the first part of the title before common separators
                title = articles[0]['title'].lower()
                for sep in [':', '-', '|', '(', 'stock', 'shares']:
                    if sep in title:
                        title = title.split(sep)[0]
                company = title.strip()
            else:
                company = symbol
        else:
            company = company_names.get(symbol)
        
        # Get top business headlines for the company
        headlines_response = requests.get(
            'https://newsapi.org/v2/top-headlines',
            params={
                'q': company,
                'category': 'business',
                'language': 'en',
                'pageSize': 50
            },
            headers={
                'X-Api-Key': NEWS_API_KEY,
                'User-Agent': 'Mozilla/5.0'
            }
        )
        headlines_response.raise_for_status()
        
        # Get detailed articles about the company
        articles_response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'qInTitle': company,  # Search company name in titles
                'q': f'"{symbol}" AND (earnings OR revenue OR guidance OR upgrade OR downgrade OR "price target" OR breakout OR surge)',
                'language': 'en',
                'sortBy': 'relevancy',
                'pageSize': 50,
                'from': from_date
            },
            headers={
                'X-Api-Key': NEWS_API_KEY,
                'User-Agent': 'Mozilla/5.0'
            }
        )
        articles_response.raise_for_status()
        
        # Combine and deduplicate articles
        all_articles = headlines_response.json()['articles'] + articles_response.json()['articles']
        seen_titles = set()
        unique_articles = []
        
        for article in all_articles:
            if article['title'] not in seen_titles:
                seen_titles.add(article['title'])
                unique_articles.append(article)
        
        # Enhanced sentiment keywords
        keywords = {
            'positive': [
                'surge', 'jump', 'gain', 'rise', 'up', 'high', 'growth', 'profit', 'beat', 'strong',
                'upgrade', 'buy', 'outperform', 'overweight', 'bullish', 'momentum', 'opportunity',
                'exceeded', 'raised', 'positive', 'confident', 'record', 'breakout', 'soar'
            ],
            'negative': [
                'fall', 'drop', 'decline', 'down', 'low', 'loss', 'miss', 'weak', 'cut', 'risk',
                'downgrade', 'sell', 'underperform', 'underweight', 'bearish', 'warning', 'concern',
                'missed', 'lowered', 'negative', 'cautious', 'disappointing'
            ]
        }
        
        sentiment_scores = []
        for article in unique_articles:
            text = f"{article['title']} {article['description']}".lower()
            pos_count = sum(1 for word in keywords['positive'] if word in text)
            neg_count = sum(1 for word in keywords['negative'] if word in text)
            sentiment_scores.append(1 if pos_count > neg_count else -1 if neg_count > pos_count else 0)
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        return {
            'symbol': symbol,
            'company': company,
            'period': f'{days} days',
            'article_count': len(unique_articles),
            'sentiment_score': avg_sentiment,
            'sentiment_label': 'Bullish' if avg_sentiment > 0.3 else 'Bearish' if avg_sentiment < -0.3 else 'Neutral',
            'recent_headlines': [
                {
                    'title': article['title'],
                    'source': article['source']['name'],
                    'date': article['publishedAt']
                }
                for article in unique_articles[:5]
            ]
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None

def get_trending_stocks():
    """Find stocks with unusual news activity"""
    # Search for stocks with sudden news interest
    response = requests.get(
        'https://newsapi.org/v2/everything',
        params={
            'q': '(stock OR shares) AND (surge OR soar OR jump OR rally OR breakout)',
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100
        },
        headers={
            'X-Api-Key': NEWS_API_KEY,
            'User-Agent': 'Mozilla/5.0'
        }
    )
    response.raise_for_status()
    
    # Extract stock symbols from headlines
    symbol_pattern = r'\b[A-Z]{1,5}\b'  # 1-5 uppercase letters
    symbol_counts = {}
    
    for article in response.json()['articles']:
        title = article['title'].upper()
        symbols = set(re.findall(symbol_pattern, title))  # Use set to count each symbol once per article
        
        for symbol in symbols:
            # Skip common words and indices
            if symbol in {'A', 'I', 'AT', 'BE', 'DO', 'IT', 'ON', 'SO', 'TV', 'US', 'SPY', 'QQQ', 'DIA', 'IWM', 'VIX'}:
                continue
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    
    # Sort by mention count
    trending = sorted(
        [(symbol, count) for symbol, count in symbol_counts.items() if count >= 2],
        key=lambda x: x[1],
        reverse=True
    )
    
    return [symbol for symbol, _ in trending]

def analyze_sentiment(symbols=None):
    """Analyze sentiment for given symbols or find trending ones"""
    if symbols is None:
        print("\nScanning for trending stocks...")
        symbols = get_trending_stocks()
        print(f"Found {len(symbols)} stocks with unusual activity")
    
    # Analyze each symbol
    analysis = {}
    for symbol in symbols:
        sentiment = get_news_sentiment(symbol)
        if sentiment:
            analysis[symbol] = sentiment
            
            print(f"\nAnalysis for {symbol} ({sentiment['company']}):")
            print(f"Sentiment: {sentiment['sentiment_label']} ({sentiment['sentiment_score']:.2f})")
            print(f"Articles found: {sentiment['article_count']}")
            
            if sentiment['recent_headlines']:
                print("\nRecent Headlines:")
                for headline in sentiment['recent_headlines'][:3]:
                    print(f"- {headline['title']}")
    
    # Save analysis
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'sentiment_analysis_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nSentiment analysis saved to {output_file}")
    return analysis

if __name__ == '__main__':
    analyze_sentiment()
