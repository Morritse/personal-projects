"""
Scan for emerging opportunities in natural gas sector
"""
import os
import json
import re
from datetime import datetime, timedelta
import requests
import yfinance as yf

NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Known large gas companies to filter out
MAJOR_PLAYERS = {
    "XOM", "CVX", "COP", "EOG", "PXD",  # Oil majors
    "EQT", "CHK", "RRC", "AR", "CNX",    # Large gas producers
    "UNG", "BOIL", "KOLD", "DGAZ"        # Major ETFs
}

def get_market_cap(symbol: str) -> float:
    """Get company market cap"""
    try:
        ticker = yf.Ticker(symbol)
        return ticker.info.get('marketCap', 0)
    except:
        return 0

def scan_gas_opportunities():
    """Find emerging opportunities in natural gas sector"""
    print("\nScanning for natural gas opportunities...")
    
    # Search for natural gas related news
    response = requests.get(
        'https://newsapi.org/v2/everything',
        params={
            'q': ('(natural gas OR LNG OR shale) AND '
                 '(penny OR microcap OR "small cap" OR emerging OR unknown OR discovery OR '
                 'surge OR soar OR jump OR rally OR breakout)'),
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
    
    # Extract potential symbols
    symbol_pattern = r'\b[A-Z]{1,5}\b'
    symbol_data = {}
    
    print("\nAnalyzing news articles...")
    for article in response.json()['articles']:
        title = article['title'].upper()
        description = article['description'].upper() if article['description'] else ''
        text = f"{title} {description}"
        
        # Find stock symbols
        symbols = set(re.findall(symbol_pattern, text))
        for symbol in symbols:
            # Skip if major player or common word
            if (symbol in MAJOR_PLAYERS or 
                len(symbol) < 2 or
                symbol in {'A', 'I', 'AT', 'BE', 'DO', 'IT', 'ON', 'SO', 'TV', 'US', 'UK', 'CEO', 'CFO'}):
                continue
            
            # Initialize symbol data
            if symbol not in symbol_data:
                # Get market cap
                market_cap = get_market_cap(symbol)
                
                # Skip if market cap > $2B
                if market_cap > 2_000_000_000:
                    continue
                
                symbol_data[symbol] = {
                    'mentions': [],
                    'market_cap': market_cap,
                    'sentiment_score': 0
                }
            
            # Add mention
            symbol_data[symbol]['mentions'].append({
                'title': article['title'],
                'source': article['source']['name'],
                'date': article['publishedAt'],
                'url': article['url']
            })
            
            # Simple sentiment analysis
            keywords = {
                'positive': [
                    'surge', 'jump', 'gain', 'rise', 'up', 'high', 'growth', 'profit', 'beat',
                    'strong', 'buy', 'bullish', 'opportunity', 'discovery', 'breakthrough'
                ],
                'negative': [
                    'fall', 'drop', 'decline', 'down', 'low', 'loss', 'miss', 'weak', 'cut',
                    'risk', 'sell', 'bearish', 'warning', 'concern', 'caution'
                ]
            }
            
            text_lower = text.lower()
            pos_count = sum(1 for word in keywords['positive'] if word in text_lower)
            neg_count = sum(1 for word in keywords['negative'] if word in text_lower)
            
            symbol_data[symbol]['sentiment_score'] += (pos_count - neg_count)
    
    # Filter and sort opportunities
    opportunities = []
    for symbol, data in symbol_data.items():
        if len(data['mentions']) >= 1:  # At least one mention
            opportunities.append({
                'symbol': symbol,
                'market_cap': data['market_cap'],
                'mention_count': len(data['mentions']),
                'sentiment_score': data['sentiment_score'],
                'recent_mentions': sorted(data['mentions'], 
                                       key=lambda x: x['date'], 
                                       reverse=True)[:3]
            })
    
    # Sort by sentiment and mentions
    opportunities.sort(key=lambda x: (x['sentiment_score'], x['mention_count']), reverse=True)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report = {
        'generated_at': datetime.now().isoformat(),
        'scan_focus': 'Natural Gas Sector',
        'opportunities': opportunities
    }
    
    filename = f'gas_opportunities_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\nFound {len(opportunities)} potential opportunities:")
    for opp in opportunities[:5]:  # Show top 5
        print(f"\n{opp['symbol']}:")
        print(f"Market Cap: ${opp['market_cap']/1e6:.1f}M")
        print(f"Mentions: {opp['mention_count']}")
        print(f"Sentiment: {opp['sentiment_score']}")
        if opp['recent_mentions']:
            print("Recent News:")
            for mention in opp['recent_mentions']:
                print(f"- {mention['title']}")
    
    print(f"\nFull report saved to {filename}")
    return opportunities

if __name__ == '__main__':
    scan_gas_opportunities()
