"""
Detailed analysis of fall flows to understand negative flow patterns.

This script:
1. Focuses on fall season patterns
2. Analyzes conditions leading to negative flows
3. Identifies potential early warning signals
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

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
    """Load fall season data with context."""
    # Load flow data
    print("\nLoading flow data...")
    flow_dir = data_dir / 'full_natural_flow'
    flow_df = pd.read_csv(flow_dir / 'SBF.csv',
                         parse_dates=['DATE TIME'],
                         index_col='DATE TIME')
    flow = pd.to_numeric(flow_df['SBF'], errors='coerce')
    
    # Create DataFrame with flow
    data = pd.DataFrame({'flow': flow})
    
    # Add date features
    data['month'] = data.index.month
    data['season'] = data.index.map(get_season)
    data['year'] = data.index.year
    data['is_negative'] = data['flow'] < 0
    
    # Add context from surrounding days
    for lag in [1, 2, 3, 7, 14, 30]:
        # Previous flows
        data[f'flow_lag_{lag}d'] = data['flow'].shift(lag)
        # Flow changes
        data[f'flow_change_{lag}d'] = data['flow'] - data[f'flow_lag_{lag}d']
        # Rolling statistics
        data[f'flow_mean_{lag}d'] = data['flow'].rolling(lag).mean()
        data[f'flow_std_{lag}d'] = data['flow'].rolling(lag).std()
    
    # Filter to fall season with 30 days context
    fall_mask = data['season'] == 'fall'
    fall_dates = data.index[fall_mask]
    context_dates = []
    for date in fall_dates:
        # Include 30 days before each fall day
        context_dates.extend(pd.date_range(date - timedelta(days=30), date))
    context_dates = sorted(set(context_dates))
    
    return data.loc[context_dates]

def analyze_fall_patterns(data: pd.DataFrame, output_dir: Path):
    """Analyze patterns leading to negative flows in fall."""
    fall_data = data[data['season'] == 'fall'].copy()
    
    # 1. Basic statistics
    print("\nFall Flow Statistics:")
    stats = {
        'total_days': len(fall_data),
        'negative_days': fall_data['is_negative'].sum(),
        'negative_pct': fall_data['is_negative'].mean() * 100,
        'min_flow': fall_data['flow'].min(),
        'median_flow': fall_data['flow'].median(),
        'mean_flow': fall_data['flow'].mean(),
        'std_flow': fall_data['flow'].std()
    }
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # 2. Analyze conditions before negative flows
    neg_dates = fall_data[fall_data['is_negative']].index
    
    # Plot flow patterns around negative events
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Flow levels leading to negative
    for days in [7, 14, 30]:
        means = []
        for date in neg_dates:
            window = pd.date_range(date - timedelta(days=days), date)
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
            prev = data.loc[date - timedelta(days=days), 'flow']
            curr = data.loc[date, 'flow']
            changes.append(curr - prev)
        axes[0, 1].hist(changes, alpha=0.5, label=f'{days}d change')
    axes[0, 1].set_title('Flow Changes Before Negative Events')
    axes[0, 1].set_xlabel('Flow Change')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Monthly distribution
    monthly_neg = fall_data.groupby('month')['is_negative'].mean() * 100
    monthly_neg.plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title('Negative Flow Frequency by Month')
    axes[1, 0].set_ylabel('Percent of Days')
    axes[1, 0].grid(True)
    
    # Year-over-year comparison
    yearly_neg = fall_data.groupby('year')['is_negative'].mean() * 100
    yearly_neg.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Negative Flow Frequency by Year')
    axes[1, 1].set_ylabel('Percent of Days')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fall_flow_patterns.png')
    plt.close()
    
    # 3. Identify potential warning signals
    warning_cols = [col for col in fall_data.columns 
                   if any(x in col for x in ['lag', 'change', 'mean', 'std'])]
    
    # Correlation with negative flows
    corr = fall_data[['is_negative'] + warning_cols].corr()['is_negative']
    
    print("\nTop warning signals (correlation with negative flows):")
    top_signals = corr.sort_values(ascending=False)
    for col, corr in top_signals.items():
        if col != 'is_negative':
            print(f"  {col}: {corr:.3f}")
    
    # Plot distributions for top signals
    top_cols = top_signals.index[1:6]  # Skip is_negative itself
    fig, axes = plt.subplots(len(top_cols), 1, figsize=(12, 4*len(top_cols)))
    
    for i, col in enumerate(top_cols):
        sns.boxplot(data=fall_data, x='is_negative', y=col, ax=axes[i])
        axes[i].set_title(f'{col} vs Negative Flows')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'warning_signals.png')
    plt.close()
    
    # 4. Extreme negative flow analysis
    extreme_neg = fall_data[fall_data['flow'] < fall_data['flow'].quantile(0.01)]
    
    print("\nExtreme Negative Flow Events:")
    for date, row in extreme_neg.iterrows():
        print(f"\nDate: {date.strftime('%Y-%m-%d')}")
        print(f"  Flow: {row['flow']:.1f}")
        print("  Prior conditions:")
        for lag in [1, 3, 7]:
            print(f"    {lag}d ago: {row[f'flow_lag_{lag}d']:.1f}")
            print(f"    {lag}d change: {row[f'flow_change_{lag}d']:.1f}")

def main():
    """Analyze fall flow patterns."""
    data_dir = Path('data/cdec')
    output_dir = data_dir / 'fall_flow_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Load and analyze data
    data = load_data(data_dir)
    analyze_fall_patterns(data, output_dir)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
