import pandas as pd
from datetime import datetime
from finvizfinance.screener.overview import Overview

class FinvizScanner:
    def get_bearish_stocks(self, limit=20):
        """
        Get bearish stocks from Finviz based on:
        - High short float (>20%)
        - Recent analyst downgrades
        - Negative price performance
        """
        try:
            print("Fetching Finviz bearish stocks...")
            
            foverview = Overview()
            
            # Set filters for bearish stocks
            filters_dict = {
                'Short Float': 'Over 20%',
                'Analyst Recom': 'Sell or worse',
                'Performance': 'Month -20% or lower'
            }
            
            foverview.set_filter(filters_dict)
            df = foverview.screener_view()
            
            if df.empty:
                return df
                
            # Sort by short float percentage
            df = df.sort_values('Short Float', ascending=False).head(limit)
            
            # Add source and timestamp
            df['source'] = 'finviz'
            df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"Found {len(df)} bearish stocks from Finviz")
            return df
            
        except Exception as e:
            print(f"Error fetching Finviz data: {str(e)}")
            return pd.DataFrame()

def main():
    scanner = FinvizScanner()
    df = scanner.get_bearish_stocks()
    
    if not df.empty:
        print("\nTop Bearish Stocks:")
        for _, row in df.iterrows():
            print(f"\n{row['ticker']} ({row['company']}):")
            print(f"Short Float: {row['short_float']}%")
            print(f"Price: ${row['price']}")
            print(f"Change: {row['change']}%")
            print(f"Volume: {row['volume']:,}")
    else:
        print("No bearish stocks found")

if __name__ == "__main__":
    main()
