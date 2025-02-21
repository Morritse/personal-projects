import os
import requests
import pandas as pd
from datetime import datetime

class WSBScanner:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_trending_stocks(self, limit=50):
        """
        Get trending stocks from ApeWisdom
        Returns DataFrame with columns: ticker, mentions, rank
        """
        try:
            print("Fetching ApeWisdom data...")
            url = 'https://apewisdom.io/api/v1.0/filter/all-stocks/page/1'
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                print(f"Error: ApeWisdom API returned status code {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            results = data.get('results', [])
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            if df.empty:
                return df
                
            # Rename and calculate columns
            df = df.rename(columns={
                'ticker': 'ticker',
                'mentions': 'mentions'
            })
            
            # Add score and sentiment (normalized rank)
            df['score'] = df['rank'].rank(pct=True)  # Percentile rank (0-1)
            df['sentiment'] = df['score']  # Same as score since ApeWisdom already factors in sentiment
            
            # Add source and timestamp
            df['source'] = 'apewisdom'
            df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get top N stocks (already ranked by ApeWisdom)
            df = df.head(limit)
            
            print(f"Found {len(df)} trending stocks from ApeWisdom")
            return df
            
        except Exception as e:
            print(f"Error fetching ApeWisdom data: {str(e)}")
            return pd.DataFrame()

def main():
    scanner = WSBScanner()
    df = scanner.get_trending_stocks()
    
    if not df.empty:
        print("\nTop Trending Stocks:")
        for _, row in df.iterrows():
            print(f"\n{row['ticker']}:")
            print(f"Mentions: {row['mentions']}")
            print(f"Rank: {row['rank']}")
    else:
        print("No trending stocks found")

if __name__ == "__main__":
    main()
