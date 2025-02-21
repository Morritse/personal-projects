import pandas as pd
import os

def fix_file_format(filepath):
    # Read CSV
    df = pd.read_csv(filepath)
    
    # Convert column names to title case
    df.columns = [col.title() for col in df.columns]
    
    # Save back to CSV
    df.to_csv(filepath, index=False)
    print(f"Fixed format for {filepath}")

def main():
    data_dir = "data"
    
    # Process all CSV files in data directory except 6E and 6J
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and not filename.startswith("6"):
            filepath = os.path.join(data_dir, filename)
            fix_file_format(filepath)

if __name__ == "__main__":
    main()
