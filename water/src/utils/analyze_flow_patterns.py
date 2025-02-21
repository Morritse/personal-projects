"""
Analyze flow patterns to understand negative flow occurrences.

This script:
1. Identifies patterns leading to negative flows
2. Analyzes seasonal and yearly trends
3. Helps understand flow regime changes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

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

def load_flow_data(data_dir: Path) -> pd.DataFrame:
    """Load and prepare flow data with context."""
    # Load flow data
    print("\nLoading flow data...")
    flow_dir = data_dir / 'full_natural_flow'
    flow_df = pd.read_csv(flow_dir / 'SBF.csv',
                         parse_dates=['DATE TIME'],
                         index_col='DATE TIME')
    
    # Create base dataframe
    data = pd.DataFrame({'flow': pd.to_numeric(flow_df['SBF'], errors='coerce')})
    
    # Add date features
    data['year'] = data.index.year
    data['month'] = data.index.month
    data['season'] = data.index.map(get_season)
    data['is_negative'] = data['flow'] < 0
    
    # Add flow context
    for window in [1, 3, 7, 14, 30]:
        # Previous flows
        data[f'flow_lag_{window}d'] = data['flow'].shift(window)
        # Flow changes
        data[f'flow_change_{window}d'] = data['flow'] - data[f'flow_lag_{window}d']
        # Rolling statistics
        data[f'flow_mean_{window}d'] = data['flow'].rolling(window).mean()
        data[f'flow_std_{window}d'] = data['flow'].rolling(window).std()
    
    return data

def analyze_negative_patterns(data: pd.DataFrame, output_dir: Path):
    """Analyze patterns around negative flows."""
    # 1. Basic statistics
    print("\nOverall Flow Statistics:")
    stats = {
        'total_days': len(data),
        'negative_days': data['is_negative'].sum(),
        'negative_pct': data['is_negative'].mean() * 100,
        'min_flow': data['flow'].min(),
        'max_flow': data['flow'].max(),
        'mean_flow': data['flow'].mean(),
        'median_flow': data['flow'].median()
    }
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # 2. Seasonal patterns
    print("\nSeasonal Statistics:")
    seasonal = data.groupby('season').agg({
        'flow': ['count', 'min', 'mean', 'std'],
        'is_negative': ['sum', 'mean']
    })
    seasonal.columns = ['days', 'min_flow', 'mean_flow', 'std_flow', 
                       'negative_days', 'negative_pct']
    seasonal['negative_pct'] *= 100
    print(seasonal)
    
    # Plot seasonal patterns
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Flow distribution by season
    sns.boxplot(data=data, x='season', y='flow', ax=axes[0, 0])
    axes[0, 0].set_title('Flow Distribution by Season')
    axes[0, 0].set_ylabel('Flow')
    axes[0, 0].grid(True)
    
    # Negative flow frequency by season
    seasonal['negative_pct'].plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Negative Flow Frequency by Season')
    axes[0, 1].set_ylabel('Percent of Days')
    axes[0, 1].grid(True)
    
    # Monthly pattern
    monthly = data.groupby('month')['is_negative'].mean() * 100
    monthly.plot(kind='line', marker='o', ax=axes[1, 0])
    axes[1, 0].set_title('Negative Flow Frequency by Month')
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Percent of Days')
    axes[1, 0].grid(True)
    
    # Yearly trend
    yearly = data.groupby('year')['is_negative'].mean() * 100
    yearly.plot(kind='line', marker='o', ax=axes[1, 1])
    axes[1, 1].set_title('Negative Flow Frequency by Year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Percent of Days')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'flow_patterns.png')
    plt.close()
    
    # 3. Analyze conditions leading to negative flows
    print("\nAnalyzing conditions before negative flows...")
    neg_dates = data[data['is_negative']].index
    
    # Plot flow patterns around negative events
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Flow levels before negative
    for days in [7, 14, 30]:
        means = []
        for date in neg_dates:
            window = pd.date_range(date - pd.Timedelta(days=days), date)
            means.append(data.loc[window, 'flow'].mean())
        axes[0, 0].hist(means, alpha=0.5, label=f'{days}d mean')
    axes[0, 0].set_title('Flow Levels Before Negative Events')
    axes[0, 0].set_xlabel('Mean Flow')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Flow changes before negative
    for days in [1, 3, 7]:
        changes = []
        for date in neg_dates:
            prev = data.loc[date - pd.Timedelta(days=days), 'flow']
            curr = data.loc[date, 'flow']
            changes.append(curr - prev)
        axes[0, 1].hist(changes, alpha=0.5, label=f'{days}d change')
    axes[0, 1].set_title('Flow Changes Before Negative Events')
    axes[0, 1].set_xlabel('Flow Change')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Flow variability
    for days in [7, 14]:
        stds = []
        for date in neg_dates:
            window = pd.date_range(date - pd.Timedelta(days=days), date)
            stds.append(data.loc[window, 'flow'].std())
        axes[1, 0].hist(stds, alpha=0.5, label=f'{days}d std')
    axes[1, 0].set_title('Flow Variability Before Negative Events')
    axes[1, 0].set_xlabel('Flow Standard Deviation')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Extreme negative events
    extreme_neg = data[data['flow'] < data['flow'].quantile(0.01)]
    if not extreme_neg.empty:
        axes[1, 1].scatter(extreme_neg.index, extreme_neg['flow'], alpha=0.7)
        axes[1, 1].set_title('Extreme Negative Flow Events')
        axes[1, 1].set_ylabel('Flow')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'negative_flow_analysis.png')
    plt.close()
    
    # 4. Identify warning signals
    print("\nAnalyzing potential warning signals...")
    lag_cols = [col for col in data.columns 
                if any(x in col for x in ['lag', 'change', 'mean', 'std'])]
    
    # Correlation with negative flows
    corr = data[['is_negative'] + lag_cols].corr()['is_negative']
    
    print("\nTop warning signals (correlation with negative flows):")
    top_signals = corr.sort_values(ascending=False)
    for col, corr in top_signals.items():
        if col != 'is_negative':
            print(f"  {col}: {corr:.3f}")
    
    # Plot distributions for top signals
    top_cols = top_signals.index[1:6]  # Skip is_negative itself
    fig, axes = plt.subplots(len(top_cols), 1, figsize=(12, 4*len(top_cols)))
    
    for i, col in enumerate(top_cols):
        sns.boxplot(data=data, x='is_negative', y=col, ax=axes[i])
        axes[i].set_title(f'{col} vs Negative Flows')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'warning_signals.png')
    plt.close()

def main():
    """Analyze flow patterns."""
    data_dir = Path('data/cdec')
    output_dir = data_dir / 'flow_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Load and analyze data
    data = load_flow_data(data_dir)
    analyze_negative_patterns(data, output_dir)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
