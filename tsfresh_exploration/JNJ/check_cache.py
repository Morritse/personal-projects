import pandas as pd

def check_cache():
    """Check what data we have in the cache files."""
    # Load cached data
    jnj_data = pd.read_pickle('data/cache/jnj_data.pkl')
    xlv_data = pd.read_pickle('data/cache/xlv_data.pkl')
    
    print("\nJNJ Data:")
    print(f"Date Range: {jnj_data.index[0]} to {jnj_data.index[-1]}")
    print("\nYearly Data Points:")
    yearly_counts = jnj_data.groupby(jnj_data.index.year).size()
    for year, count in yearly_counts.items():
        print(f"{year}: {count} hours")
    
    print("\nXLV Data:")
    print(f"Date Range: {xlv_data.index[0]} to {xlv_data.index[-1]}")
    print("\nYearly Data Points:")
    yearly_counts = xlv_data.groupby(xlv_data.index.year).size()
    for year, count in yearly_counts.items():
        print(f"{year}: {count} hours")

if __name__ == "__main__":
    check_cache()
