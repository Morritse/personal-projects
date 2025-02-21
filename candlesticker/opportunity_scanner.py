"""
Scan for emerging 'hype' or lesser-known stock opportunities using NewsAPI.
Focus on microcap/penny references, skip major tickers.
"""

import os
import json
import re
from datetime import datetime
import requests
from typing import Optional, List, Dict, Any

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
if not NEWS_API_KEY:
    raise ValueError("Please set the NEWS_API_KEY environment variable.")

# Common words to filter out (articles, prepositions, etc.)
COMMON_WORDS = {
    # Articles/Conjunctions
    'THE', 'AND', 'FOR', 'BUT', 'WITH', 'OR', 'NOR', 'YET', 'SO',
    # Prepositions
    'IN', 'ON', 'AT', 'TO', 'OF', 'BY', 'AS', 'INTO', 'ONTO', 'UPON',
    'FROM', 'OUT', 'OVER', 'UNDER', 'AFTER', 'BEFORE', 'DURING',
    # Verbs
    'IS', 'ARE', 'WAS', 'WERE', 'BE', 'BEEN', 'BEING',
    'HAVE', 'HAS', 'HAD', 'DO', 'DOES', 'DID',
    'WILL', 'WOULD', 'CAN', 'COULD', 'MAY', 'MIGHT',
    'MAKE', 'MADE', 'TAKE', 'TOOK', 'GET', 'GOT',
    # Business/Market Terms
    'CEO', 'CFO', 'COO', 'CTO', 'IPO', 'ICO', 'NYSE', 'NASDAQ',
    'NEWS', 'STOCK', 'SHARE', 'PRICE', 'TRADE', 'MARKET', 'INDEX',
    'YEAR', 'TIME', 'DATA', 'REPORT', 'UPDATE', 'GAINS', 'LOSS',
    'BUY', 'SELL', 'HOLD', 'TOP', 'HIGH', 'LOW', 'FAST', 'SLOW',
    'BEST', 'TECH', 'TODAY', 'ITS', 'ONE', 'PER', 'AHEAD', 'AI',
    # Common Acronyms
    'USA', 'UK', 'EU', 'UN', 'GDP', 'CPI', 'FED', 'SEC', 'FDA',
    # Days/Months
    'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN',
    'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC',
    # Additional Common Words
    'NOW', 'NEW', 'NEXT', 'LAST', 'THIS', 'THAT', 'THESE', 'THOSE',
    'HERE', 'THERE', 'WHAT', 'WHEN', 'WHERE', 'WHY', 'HOW',
    'ALL', 'ANY', 'SOME', 'MANY', 'MUCH', 'MORE', 'MOST',
    'OTHER', 'ANOTHER', 'SUCH', 'SAME', 'THAN',
    'YES', 'NO', 'NOT', 'ABOUT', 'LIKE', 'JUST', 'VERY',
    # Common Verbs in Headlines
    'SAYS', 'SEES', 'PLANS', 'WANTS', 'NEEDS', 'GETS', 'GOES',
    'COMES', 'LOOKS', 'FEELS', 'SEEMS', 'SHOWS', 'GIVES',
    # Common Nouns in Headlines
    'EARLY', 'SINCE', 'CALLS', 'SEEN', 'DIGIT', 'PENNY', 'WEEK',
    'YEAR', 'YEARS', 'MONTH', 'DAY', 'TIME', 'WAY', 'THING', 'PART',
    'CASE', 'POINT', 'FACT', 'GROUP', 'LINE', 'END', 'SIDE', 'NOTE',
    'LIST', 'TYPE', 'KIND', 'FORM', 'CAR', 'CAP', 'SET', 'ETF',
    'ALERT', 'NEWS', 'UPDATE', 'BRIEF', 'WATCH', 'FOCUS', 'PICK',
    # Exchange Names
    'NYSE', 'NASDAQ', 'AMEX', 'ASX', 'LSE', 'TSX', 'JSE', 'SSE',
    # Common Adjectives
    'GOOD', 'BAD', 'BIG', 'SMALL', 'OLD', 'YOUNG', 'LONG', 'SHORT',
    'FULL', 'EMPTY', 'HARD', 'SOFT', 'FAST', 'SLOW', 'EARLY', 'LATE',
    # Finance-specific common words
    'RATE', 'DEBT', 'FUND', 'BANK', 'CASH', 'COST', 'RISK', 'DEAL',
    'FIRM', 'PLAN', 'SALE', 'TEAM', 'LEAD', 'HEAD', 'UNIT', 'CHIEF',
    # Additional Finance Terms
    'BOND', 'EARN', 'PAID', 'SAVE', 'SPEND', 'GREW', 'GROW', 'FALL',
    'FELL', 'RISE', 'ROSE', 'GAIN', 'LOST', 'BEAT', 'MISS', 'MEET',
    'SEES', 'VIEW', 'LOOK', 'READ', 'SAYS', 'TOLD', 'TALK', 'HEAR'
}

# Major tickers to filter out (we want smaller or lesser-known)
MAJOR_TICKERS = {
    # Tech Giants
    'AAPL','MSFT','GOOG','GOOGL','META','AMZN','NVDA','TSLA',
    # Finance
    'JPM','BAC','WFC','GS','MS','V','MA','AXP','BLK','C',
    # Healthcare
    'JNJ','UNH','PFE','MRK','ABBV','AMGN','LLY','BMY','TMO',
    # Industrial/Retail
    'WMT','HD','CAT','BA','GE','MMM','HON','UPS','RTX',
    # ETFs
    'SPY','QQQ','DIA','IWM','VTI','VOO','VEA','VWO','BND',
    # Other Large Caps
    'XOM','CVX','PG','KO','PEP','CSCO','ORCL','CRM','ADBE',
    'DIS','NFLX','INTC','AMD','QCOM','TXN','IBM','UBER','ABNB'
}

def is_excluded_symbol(symbol: str) -> bool:
    """Check if symbol should be excluded based on filters"""
    return (
        symbol in COMMON_WORDS or 
        symbol in MAJOR_TICKERS or 
        len(symbol) < 3 or  # Most real tickers are 3+ chars
        symbol.isdigit() or  # Exclude pure numbers
        not symbol.isalpha() or  # Must be all letters
        any(word in symbol for word in ['ALERT', 'NEWS', 'UPDATE'])  # Common headline words
    )

def looks_like_stock_symbol(text: str, symbol: str) -> bool:
    """
    Check if symbol likely references a stock ticker based on 
    surrounding context in 'text'.
    """
    # Basic filters
    if is_excluded_symbol(symbol):
        return False
        
    # Must have at least one strong stock indicator
    strong_indicators = [
        rf'\${symbol}\b',                    # $TICK
        rf'\b{symbol}:\s',                   # TICK:
        rf'\b{symbol}\s+(?:stock|shares)\b', # TICK stock/shares
        rf'(?:ticker|symbol)\s+{symbol}\b',  # ticker/symbol TICK
        rf'{symbol}\s+(?:Inc|Corp|Ltd|Company|plc)\b',  # TICK Inc/Corp
        rf'(?:shares\s+(?:of|in))\s+{symbol}\b',  # shares of/in TICK
        rf'(?:buy|sell)\s+{symbol}\b',       # buy/sell TICK
        rf'\({symbol}\)',                    # (TICK)
        rf'\b{symbol}\s+shares\s+(?:up|down|rise|fall|jump|drop)',  # TICK shares up/down
    ]
    
    has_strong_indicator = any(re.search(pattern, text, re.IGNORECASE) 
                             for pattern in strong_indicators)
    
    if not has_strong_indicator:
        return False
        
    # Additional context check - must have stock-related words nearby
    stock_context = [
        'stock', 'share', 'market', 'trading', 'investor', 'price',
        'nasdaq', 'nyse', 'otc', 'exchange', 'broker', 'dividend',
        'earnings', 'revenue', 'profit', 'guidance', 'analyst',
        'ticker', 'securities', 'investment', 'portfolio', 'microcap',
        'penny stock', 'small cap', 'undervalued', 'breakout'
    ]
    
    # Look for stock context within reasonable distance
    text_lower = text.lower()
    symbol_pos = text_lower.find(symbol.lower())
    if symbol_pos == -1:
        return False
        
    # Check 100 chars before and after symbol
    context_start = max(0, symbol_pos - 100)
    context_end = min(len(text_lower), symbol_pos + 100)
    context = text_lower[context_start:context_end]
    
    has_stock_context = any(word in context for word in stock_context)
    if not has_stock_context:
        return False
    
    # Patterns that hint it's a real ticker mention
    context_patterns = [
        rf'\${symbol}\b',
        rf'\b{symbol}:\s',  # e.g. ABC:
        rf'\b{symbol}\s+(?:stock|shares)\b',
        rf'(?:buy|sell|trade)\s+{symbol}\b',
        rf'{symbol}\s+(?:up|down|gains|falls|rises|drops|jumps|plunges)',
        rf'{symbol}\s+(?:price|market|trading|volume|float|short)',
        rf'(?:\$|\d+(?:\.\d{{2}})?)\s*{symbol}\b',
        rf'{symbol}\s+(?:inc|corp|ltd|company|plc|technologies|therapeutics|pharmaceuticals)\b',
        rf'(?:shares\s+(?:of|in))\s+{symbol}\b',
        rf'(?:ticker|symbol)\s+{symbol}\b',
        rf'{symbol}\s+(?:rally|surge|soar|plunge|crash|squeeze)',
        rf'{symbol}\s+(?:earnings|revenue|profit|guidance|eps)',
        rf'(?:analyst|upgrade|downgrade|rating)\s+{symbol}',
        rf'(?:bullish|bearish)\s+on\s+{symbol}\b',
        rf'{symbol}\s+(?:stock|shares)\s+(?:are|is|have|has)',
        rf'position\s+in\s+{symbol}\b',
        rf'{symbol}\s+(?:penny\s+stock|microcap|otc|pink\s+sheets)',
        rf'(?:penny\s+stock|microcap|otc).*{symbol}'
    ]
    
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in context_patterns)

def fetch_articles(query: str) -> List[Dict[str, Any]]:
    """Fetch articles from NewsAPI with error handling"""
    try:
        response = requests.get(
            'https://newsapi.org/v2/everything',
            params={
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50
            },
            headers={
                'X-Api-Key': NEWS_API_KEY,
                'User-Agent': 'Mozilla/5.0'
            },
            timeout=10  # Add timeout
        )
        response.raise_for_status()
        return response.json()['articles']
    except requests.exceptions.Timeout:
        print(f"Timeout fetching articles for query: {query}")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return []

def scan_opportunities():
    """
    Find 'hype' or lesser-known stock opportunities from 
    penny/microcap angle.
    """
    print("\nScanning for 'hype' microcap/penny opportunities...")

    try:
        # 1) Focus on penny/microcap/hype keywords
        query1 = (
            '(("penny stock" OR microcap OR "small cap" OR "otc stock" OR "pink sheets") AND '
            '(surge OR soar OR jump OR rally OR breakout OR "massive potential" OR '
            '"hidden gem" OR "next big" OR undervalued OR "buy rating"))'
        )
        
        # 2) Additional search: emerging or disruptive small companies
        query2 = (
            '((emerging company OR "new breakthrough" OR "disruptive" OR "hidden gem") AND '
            '(fda OR patent OR trial OR partnership OR contract OR acquisition OR "short squeeze" OR spike))'
        )

        # Fetch articles
        articles = []
        articles.extend(fetch_articles(query1))
        articles.extend(fetch_articles(query2))
        
        if not articles:
            print("No articles found. Check your API key and internet connection.")
            return

        print(f"Fetched {len(articles)} total articles from both queries.\n")

        symbol_pattern = r'\b[A-Z]{2,5}\b'
        symbol_data = {}

        print("Analyzing news articles for potential tickers...\n")
        for article in articles:
            title = (article.get('title') or '').upper()
            desc = (article.get('description') or '').upper()
            text = f"{title} {desc}"

            # Search for ticker-like patterns
            possible_symbols = set(re.findall(symbol_pattern, text))
            for sym in possible_symbols:
                # Double-check exclusions (sometimes case sensitivity matters)
                if is_excluded_symbol(sym.upper()) or not looks_like_stock_symbol(text, sym):
                    continue

                if sym not in symbol_data:
                    symbol_data[sym] = {
                        'mentions': [],
                        'sentiment_score': 0,
                        'catalysts': set()
                    }
                
                symbol_data[sym]['mentions'].append({
                    'title': article.get('title'),
                    'source': article['source'].get('name'),
                    'date': article.get('publishedAt'),
                    'url': article.get('url')
                })

                # Quick sentiment approach
                pos_words = [
                    'surge', 'jump', 'gain', 'rise', 'up', 'growth', 'profit', 'beat', 
                    'strong', 'buy', 'bullish', 'opportunity', 'promising', 'successful',
                    'breakthrough', 'innovative', 'leading', 'massive', 'soar', 'rebound'
                ]
                neg_words = [
                    'fall', 'drop', 'decline', 'down', 'loss', 'warning', 'risk', 
                    'bearish', 'caution', 'lawsuit', 'scandal', 'collapse', 'fail'
                ]
                lower_text = text.lower()
                pos_count = sum(1 for w in pos_words if w in lower_text)
                neg_count = sum(1 for w in neg_words if w in lower_text)
                symbol_data[sym]['sentiment_score'] += (pos_count - neg_count)

                # Catalyst keywords
                catalyst_map = {
                    'partnership': ['partnership','collaboration','deal','agreement'],
                    'fda': ['fda','approval','clinical','drug','phase'],
                    'acquisition': ['acquire','acquisition','merger','buyout'],
                    'patent': ['patent','license'],
                    'short_squeeze': ['short squeeze','short interest'],
                    'breakout': ['breakout','skyrocket','spike']
                }
                for cat, words in catalyst_map.items():
                    if any(word in lower_text for word in words):
                        symbol_data[sym]['catalysts'].add(cat)

        # Build final list
        hype_list = []
        for sym, data in symbol_data.items():
            mention_count = len(data['mentions'])
            # We only keep if mention_count >= 1 (since it's lesser known, might not appear often)
            if mention_count >= 1:
                # Filter out mentions with no date before sorting
                valid_mentions = [m for m in data['mentions'] if m['date']]
                sorted_mentions = sorted(valid_mentions, key=lambda x: x['date'], reverse=True)
                
                hype_list.append({
                    'symbol': sym,
                    'mention_count': mention_count,
                    'sentiment_score': data['sentiment_score'],
                    'catalysts': list(data['catalysts']),
                    'recent_mentions': sorted_mentions[:3]
                })

        # Sort by sentiment + mention_count
        hype_list.sort(key=lambda x: (x['sentiment_score'], x['mention_count']), reverse=True)

        # Save results
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report = {
            'generated_at': datetime.now().isoformat(),
            'focus': 'Hype or lesser-known microcaps',
            'results': hype_list
        }

        filename = f'hype_opportunities_{stamp}.json'
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nFound {len(hype_list)} potential smaller-cap 'hype' picks. Top 10:\n")
        for item in hype_list[:10]:
            print(f"Symbol: {item['symbol']} | Mentions: {item['mention_count']} | Sentiment: {item['sentiment_score']}")
            if item['catalysts']:
                print(f"  Catalysts: {', '.join(item['catalysts'])}")
            if item['recent_mentions']:
                print(f"  Headlines:")
                for mention in item['recent_mentions'][:2]:
                    print(f"   - {mention['title']}  ({mention['source']})")

        print(f"\nReport saved as {filename}")

    except Exception as e:
        print(f"\nError during scan: {str(e)}")

def main():
    try:
        scan_opportunities()
    except KeyboardInterrupt:
        print("\nScan interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        print("\nScan complete.")

if __name__ == '__main__':
    main()
