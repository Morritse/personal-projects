"""
Analyze patterns and causes of negative flows.

This script:
1. Identifies when negative flows occur
2. Analyzes relationships with other variables
3. Visualizes seasonal patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def get_season(date: pd.Timestamp) -> str:
    """Get season for a given date."""
    month = date.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

def load_data(data_dir: Path) -> pd.DataFrame:
    """Load flow and related data."""
    # Load flow data
    print("\nLoading flow data...")
    flow_dir = data_dir / 'full_natural_flow'
    flow_df = pd.read_csv(flow_dir / 'SBF.csv',
                         parse_dates=['DATE TIME'],
                         index_col='DATE TIME')
    flow = pd.to_numeric(flow_df['SBF'], errors='coerce')
    
    # Load storage data (might explain negative flows)
    print("\nLoading storage data...")
    storage_data = []
    storage_dir = data_dir / 'storage'
    if storage_dir.exists():
        for station_file in storage_dir.glob('*.csv'):
            df = pd.read_csv(station_file,
                           parse_dates=['DATE TIME'],
                           index_col='DATE TIME')
            station_id = station_file.stem
            df[f"storage_{station_id}"] = pd.to_numeric(df[station_id], errors='coerce')
            storage_data.append(df)
    
    # Combine data
    data = pd.DataFrame({'flow': flow})
    if storage_data:
        storage_df = pd.concat(storage_data, axis=1)
        data = pd.concat([data, storage_df], axis=1)
    
    # Add date features
    data['month'] = data.index.month
    data['season'] = data.index.map(get_season)
    data['year'] = data.index.year
    data['is_negative'] = data['flow'] < 0
    
    return data

def analyze_negative_flows(data: pd.DataFrame, output_dir: Path):
    """Analyze patterns in negative flows."""
    # 1. Basic statistics
    print("\nNegative Flow Statistics:")
    neg_stats = {
        'total_count': (data['flow'] < 0).sum(),
        'total_percent': (data['flow'] < 0).mean() * 100,
        'min_value': data[data['flow'] < 0]['flow'].min(),
        'mean_value': data[data['flow'] < 0]['flow'].mean(),
        'median_value': data[data['flow'] < 0]['flow'].median()
    }
    for key, value in neg_stats.items():
        print(f"  {key}: {value:.2f}")
    
    # 2. Seasonal patterns
    seasonal_stats = data.groupby('season').agg({
        'is_negative': ['count', 'sum', 'mean'],
        'flow': ['min', 'mean', 'std']
    })
    seasonal_stats.columns = ['total_days', 'negative_days', 'negative_pct', 
                            'min_flow', 'mean_flow', 'std_flow']
    seasonal_stats['negative_pct'] *= 100
    
    print("\nSeasonal Statistics:")
    print(seasonal_stats)
    
    # 3. Plot seasonal patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Negative flow frequency by season
    seasonal_stats['negative_pct'].plot(kind='bar', ax=axes[0, 0])
    axes[0, 0].set_title('Percentage of Negative Flows by Season')
    axes[0, 0].set_ylabel('Percent of Days')
    axes[0, 0].grid(True)
    
    # Flow distribution by season
    sns.boxplot(data=data, x='season', y='flow', ax=axes[0, 1])
    axes[0, 1].set_title('Flow Distribution by Season')
    axes[0, 1].set_ylabel('Flow')
    axes[0, 1].grid(True)
    
    # Monthly pattern
    monthly_neg = data.groupby('month')['is_negative'].mean() * 100
    monthly_neg.plot(kind='line', marker='o', ax=axes[1, 0])
    axes[1, 0].set_title('Negative Flow Frequency by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Percent of Days')
    axes[1, 0].grid(True)
    
    # Time series of negative flows
    neg_flows = data[data['flow'] < 0]['flow']
    if not neg_flows.empty:
        neg_flows.plot(style='.', alpha=0.5, ax=axes[1, 1])
        axes[1, 1].set_title('Negative Flow Events Over Time')
        axes[1, 1].set_ylabel('Flow')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'negative_flow_patterns.png')
    plt.close()
    
    # 4. Storage relationship
    storage_cols = [col for col in data.columns if 'storage' in col.lower()]
    if storage_cols:
        # Calculate and add storage changes to main dataframe
        for col in storage_cols:
            data[f"{col}_change"] = data[col].diff()
        
        # Correlation analysis
        change_cols = [col + '_change' for col in storage_cols]
        analysis_cols = ['flow', 'is_negative'] + storage_cols + change_cols
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(data[analysis_cols].corr(), 
                   annot=True, 
                   cmap='RdBu', 
                   center=0,
                   fmt='.2f')
        plt.title('Correlation between Flow and Storage')
        plt.tight_layout()
        plt.savefig(output_dir / 'flow_storage_correlation.png')
        plt.close()
        
        # Plot relationships
        fig, axes = plt.subplots(len(storage_cols), 2,
                               figsize=(15, 5*len(storage_cols)))
        
        for i, col in enumerate(storage_cols):
            # Storage levels
            sns.scatterplot(data=data,
                          x=col,
                          y='flow',
                          hue='season',
                          alpha=0.5,
                          ax=axes[i, 0])
            axes[i, 0].set_title(f'Flow vs {col}')
            axes[i, 0].grid(True)
            
            # Storage changes
            change_col = f"{col}_change"
            sns.scatterplot(data=data,
                          x=change_col,
                          y='flow',
                          hue='season',
                          alpha=0.5,
                          ax=axes[i, 1])
            axes[i, 1].set_title(f'Flow vs {col} Changes')
            axes[i, 1].grid(True)
            
            # Highlight negative flows
            neg_mask = data['flow'] < 0
            if neg_mask.any():
                axes[i, 0].scatter(data.loc[neg_mask, col],
                                 data.loc[neg_mask, 'flow'],
                                 color='red',
                                 alpha=0.7,
                                 label='Negative Flow')
                axes[i, 1].scatter(data.loc[neg_mask, change_col],
                                 data.loc[neg_mask, 'flow'],
                                 color='red',
                                 alpha=0.7,
                                 label='Negative Flow')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'flow_vs_storage_changes.png')
        plt.close()

def main():
    """Analyze negative flow patterns."""
    data_dir = Path('data/cdec')
    output_dir = data_dir / 'negative_flow_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Load and analyze data
    data = load_data(data_dir)
    analyze_negative_flows(data, output_dir)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
