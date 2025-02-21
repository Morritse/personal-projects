import pandas as pd
from wsb_scanner import WSBScanner
from stocktwits_scanner import StocktwitsScanner
from technical_analysis import TechnicalAnalyzer

class SocialScanner:
    def __init__(self):
        self.wsb_scanner = WSBScanner()
        self.stocktwits_scanner = StocktwitsScanner()
        self.technical_analyzer = TechnicalAnalyzer()
    
    def get_trending_stocks(self, limit=20):
        """Get trending stocks from Reddit and Stocktwits"""
        # Get data from sources
        wsb_df = self.wsb_scanner.get_trending_stocks(limit)
        if not wsb_df.empty:
            wsb_df = wsb_df.head(limit)
            wsb_df['ape_rank'] = range(1, len(wsb_df) + 1)
            print(f"Found {len(wsb_df)} trending stocks from ApeWisdom")
        
        print("\nFetching Stocktwits data...")
        st_df = self.stocktwits_scanner.get_trending_stocks(limit)
        st_df = st_df.head(limit)
        st_df['st_rank'] = range(1, len(st_df) + 1)
        print(f"Found {len(st_df)} trending stocks from Stocktwits")
        
        # Combine dataframes
        if not wsb_df.empty and not st_df.empty:
            # Merge on ticker
            df = pd.merge(wsb_df[['ticker', 'ape_rank']], 
                         st_df[['ticker', 'st_rank']], 
                         on='ticker', 
                         how='outer')
            
            # Fill missing ranks
            df['ape_rank'] = df['ape_rank'].fillna(limit + 1)
            df['st_rank'] = df['st_rank'].fillna(limit + 1)
            
            # Calculate sentiment rank
            df['in_both'] = (~df['ape_rank'].isna()) & (~df['st_rank'].isna())
            df['sentiment_rank'] = df.apply(
                lambda row: (row['ape_rank'] + row['st_rank']) / 2 if row['in_both']
                else min(row['ape_rank'], row['st_rank']),
                axis=1
            )
            
            # Run technical analysis silently
            ta_results = []
            for _, row in df.iterrows():
                ta_data = self.technical_analyzer.analyze_stock(row['ticker'])
                if ta_data:
                    ta_results.append({
                        'ticker': row['ticker'],
                        'technical_score': ta_data['score']
                    })
            
            # Add technical ranks
            ta_df = pd.DataFrame(ta_results)
            if not ta_df.empty:
                ta_df = ta_df.sort_values('technical_score', ascending=False)
                ta_df['ta_rank'] = range(1, len(ta_df) + 1)
                
                # Merge technical ranks
                df = pd.merge(df, ta_df[['ticker', 'ta_rank']], on='ticker', how='left')
                
                # Calculate final rank
                df['final_rank'] = df['sentiment_rank'] + df['ta_rank']
                return df.sort_values('final_rank')
            
        print("No stocks found or no technical data available")
        return pd.DataFrame()

def main():
    scanner = SocialScanner()
    df = scanner.get_trending_stocks()
    
    if not df.empty:
        print("\nTop 10 Ranked Stocks:")
        for _, row in df.head(10).iterrows():
            print(f"{row['ticker']}: Rank {row['final_rank']:.1f} (Sentiment: {row['sentiment_rank']:.1f}, TA: {row['ta_rank']:.0f})")
    else:
        print("No stocks found")

if __name__ == "__main__":
    main()
