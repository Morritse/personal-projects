"""
Analyze how feature relationships change across seasons.

This script:
1. Groups features by type (snow, temp, precip)
2. Analyzes their importance by season
3. Shows how relationships change seasonally
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

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
    """Load and prepare sensor and flow data."""
    # Load flow data
    flow_dir = data_dir / 'full_natural_flow'
    flow_df = pd.read_csv(flow_dir / 'SBF.csv',
                         parse_dates=['DATE TIME'],
                         index_col='DATE TIME')
    flow = pd.to_numeric(flow_df['SBF'], errors='coerce')
    
    # Load sensor data by type
    sensor_types = {
        'snow': ['snow_water_content', 'snow_depth'],
        'temp': ['temperature_air_average', 'temperature_air_maximum', 'temperature_air_minimum'],
        'precip': ['precipitation'],
        'storage': ['storage']
    }
    
    sensor_data = {}
    for category, types in sensor_types.items():
        category_data = []
        for sensor_type in types:
            sensor_dir = data_dir / sensor_type
            if not sensor_dir.exists():
                continue
                
            print(f"\nLoading {sensor_type} data...")
            for station_file in sensor_dir.glob('*.csv'):
                df = pd.read_csv(station_file,
                               parse_dates=['DATE TIME'],
                               index_col='DATE TIME')
                station_id = station_file.stem
                df[station_id] = pd.to_numeric(df[station_id], errors='coerce')
                if not df.empty:
                    category_data.append(df)
                    print(f"Loaded station: {station_id}")
        
        if category_data:
            sensor_data[category] = pd.concat(category_data, axis=1)
    
    # Combine all data
    all_data = pd.concat([flow.rename('flow')] + 
                        [df for df in sensor_data.values()],
                        axis=1)
    
    # Add season labels
    all_data['season'] = all_data.index.map(get_season)
    
    return all_data

def analyze_flow_ranges(data: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Analyze how flow ranges and distributions vary by season."""
    seasonal_flows = []
    for season in ['winter', 'spring', 'summer', 'fall']:
        season_data = data[data['season'] == season]['flow']
        stats = {
            'season': season,
            'min': season_data.min(),
            'q25': season_data.quantile(0.25),
            'median': season_data.median(),
            'q75': season_data.quantile(0.75),
            'max': season_data.max(),
            'negative_pct': (season_data < 0).mean() * 100
        }
        seasonal_flows.append(stats)
    
    # Create subplots for raw and normalized flows
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    stats_df = pd.DataFrame(seasonal_flows)
    
    # Raw flow box plot
    ax1.boxplot([data[data['season'] == season]['flow'] 
                for season in ['winter', 'spring', 'summer', 'fall']],
                labels=['Winter', 'Spring', 'Summer', 'Fall'])
    ax1.set_title('Raw Flow Distribution by Season')
    ax1.set_ylabel('Flow')
    ax1.grid(True, alpha=0.3)
    
    # Add text with negative flow percentages
    for i, row in stats_df.iterrows():
        if row['negative_pct'] > 0:
            ax1.text(i+1, row['min'], 
                    f"{row['negative_pct']:.1f}% neg",
                    ha='center', va='top')
    
    # Normalized flow box plot (by season)
    normalized_flows = []
    for season in ['winter', 'spring', 'summer', 'fall']:
        season_data = data[data['season'] == season]['flow']
        season_mean = season_data.mean()
        season_std = season_data.std()
        normalized = (season_data - season_mean) / season_std
        normalized_flows.append(normalized)
    
    ax2.boxplot(normalized_flows,
                labels=['Winter', 'Spring', 'Summer', 'Fall'])
    ax2.set_title('Normalized Flow Distribution by Season')
    ax2.set_ylabel('Standard Deviations from Mean')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'flow_distributions.png')
    plt.close()
    
    return stats_df

def analyze_feature_transitions(data: pd.DataFrame, output_dir: Path):
    """Analyze how important features transition between seasons."""
    # Calculate normalized values by season for each feature type
    feature_means = {}
    for feature_type in ['snow', 'temperature', 'precipitation']:
        cols = [col for col in data.columns if feature_type in col.lower()]
        if cols:
            # Normalize each column first
            normalized_data = data[cols].copy()
            for col in cols:
                col_data = normalized_data[col]
                normalized_data[col] = (col_data - col_data.mean()) / col_data.std()
            
            seasonal_means = []
            for season in ['winter', 'spring', 'summer', 'fall']:
                season_data = normalized_data[data['season'] == season]
                mean_value = season_data.mean().mean()
                seasonal_means.append({
                    'season': season,
                    'value': mean_value,
                    'feature_type': feature_type
                })
            feature_means[feature_type] = seasonal_means
    
    # Plot transitions
    if feature_means:
        plt.figure(figsize=(12, 6))
        
        for feature_type, means in feature_means.items():
            df = pd.DataFrame(means)
            plt.plot(df['season'], df['value'], 
                    marker='o', label=feature_type.title(),
                    linewidth=2, markersize=8)
        
        plt.title('Feature Type Transitions Across Seasons')
        plt.ylabel('Mean Value (Normalized)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_transitions.png')
        plt.close()

def analyze_correlations(data: pd.DataFrame, output_dir: Path):
    """Analyze how correlations change by season.
    
    Shows how the relationship between features and flow varies across seasons.
    For example:
    - Snow features may correlate strongly in spring (melting)
    - Temperature more important in summer
    - Precipitation matters most in winter
    """
    # Group features
    feature_groups = {
        'Snow': [col for col in data.columns if 'snow' in col.lower()],
        'Temperature': [col for col in data.columns if 'temperature' in col.lower()],
        'Precipitation': [col for col in data.columns if 'precipitation' in col.lower()],
        'Storage': [col for col in data.columns if 'storage' in col.lower()]
    }
    
    # Calculate correlations by season
    seasonal_corr = {}
    for season in ['winter', 'spring', 'summer', 'fall']:
        season_data = data[data['season'] == season].drop('season', axis=1)
        
        # Calculate correlations with flow
        corr = season_data.corr()['flow'].drop('flow')
        
        # Group by feature type
        grouped_corr = {}
        for group, cols in feature_groups.items():
            group_corr = corr[corr.index.isin(cols)]
            if not group_corr.empty:
                grouped_corr[group] = group_corr
        
        seasonal_corr[season] = grouped_corr
    
    # Plot correlation heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    axes = axes.ravel()
    
    for i, (season, corr_dict) in enumerate(seasonal_corr.items()):
        # Combine correlations if any exist
        corr_dfs = [df for df in corr_dict.values() if not df.empty]
        if corr_dfs:
            all_corr = pd.concat(corr_dfs)
            
            # Create heatmap
            sns.heatmap(all_corr.to_frame(),
                       cmap='RdBu',
                       center=0,
                       vmin=-1,
                       vmax=1,
                       ax=axes[i])
            axes[i].set_title(f'{season.title()} Correlations with Flow')
        else:
            axes[i].text(0.5, 0.5, 'No correlations found',
                        horizontalalignment='center',
                        verticalalignment='center')
            axes[i].set_title(f'{season.title()} - No Data')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'seasonal_correlations.png')
    plt.close()

def analyze_feature_importance(data: pd.DataFrame, output_dir: Path):
    """Analyze how different features contribute by season.
    
    Demonstrates why seasonal models are valuable:
    1. Different features matter in each season
    2. Flow ranges vary dramatically
    3. Feature relationships change seasonally
    """
    # Group features
    feature_groups = {
        'Snow Features': len([col for col in data.columns if 'snow' in col.lower()]),
        'Temperature Features': len([col for col in data.columns if 'temperature' in col.lower()]),
        'Precipitation Features': len([col for col in data.columns if 'precipitation' in col.lower()]),
        'Storage Features': len([col for col in data.columns if 'storage' in col.lower()])
    }
    
    # Plot feature counts with better formatting
    plt.figure(figsize=(10, 5))
    ax = pd.Series(feature_groups).plot(kind='bar')
    plt.title('Available Features by Type')
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    # Add value labels on bars
    for i, v in enumerate(feature_groups.values()):
        ax.text(i, v, str(v), ha='center', va='bottom')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_counts.png')
    plt.close()
    
    # Analyze seasonal patterns
    seasonal_stats = []
    for season in ['winter', 'spring', 'summer', 'fall']:
        season_data = data[data['season'] == season]
        
        # Calculate statistics
        snow_cols = [col for col in data.columns if 'snow' in col.lower()]
        temp_cols = [col for col in data.columns if 'temperature' in col.lower()]
        precip_cols = [col for col in data.columns if 'precipitation' in col.lower()]
        
        stats = {
            'season': season,
            'mean_flow': season_data['flow'].mean(),
            'std_flow': season_data['flow'].std(),
        }
        
        try:
            # Add normalized feature type means if columns exist
            if snow_cols:
                snow_data = season_data[snow_cols].copy()
                for col in snow_cols:
                    snow_data[col] = (snow_data[col] - data[col].mean()) / data[col].std()
                stats['snow_importance'] = snow_data.mean().mean()
            if temp_cols:
                temp_data = season_data[temp_cols].copy()
                for col in temp_cols:
                    temp_data[col] = (temp_data[col] - data[col].mean()) / data[col].std()
                stats['temp_importance'] = temp_data.mean().mean()
            if precip_cols:
                precip_data = season_data[precip_cols].copy()
                for col in precip_cols:
                    precip_data[col] = (precip_data[col] - data[col].mean()) / data[col].std()
                stats['precip_importance'] = precip_data.mean().mean()
        except Exception as e:
            print(f"Warning: Error calculating means for {season}: {str(e)}")
        
        seasonal_stats.append(stats)
    
    if not seasonal_stats:
        print("Warning: No seasonal statistics calculated")
        return
        
    # Plot seasonal patterns
    stats_df = pd.DataFrame(seasonal_stats)
    
    # Create plots
    try:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Scale flow values relative to maximum
        max_flow = max(stats_df['mean_flow'].max(), stats_df['std_flow'].max())
        stats_df['mean_flow_scaled'] = stats_df['mean_flow'] / max_flow
        stats_df['std_flow_scaled'] = stats_df['std_flow'] / max_flow
        
        # Flow patterns with better formatting
        colors = ['#2ecc71', '#e74c3c']  # Green for mean, red for std
        ax0 = axes[0]
        stats_df.plot(x='season', y=['mean_flow_scaled', 'std_flow_scaled'],
                     kind='bar', ax=ax0, color=colors)
        ax0.set_title('Relative Flow Patterns by Season')
        ax0.set_ylabel('Relative Flow (scaled)')
        ax0.grid(True, alpha=0.3)
        
        # Add actual value labels
        for i, row in stats_df.iterrows():
            ax0.text(i, row['mean_flow_scaled'], f"{row['mean_flow']:.0f}",
                    ha='center', va='bottom')
            ax0.text(i, row['std_flow_scaled'], f"{row['std_flow']:.0f}",
                    ha='center', va='bottom')
        
        # Feature importance with better visualization
        importance_cols = [col for col in ['snow_importance', 'temp_importance', 'precip_importance']
                         if col in stats_df.columns]
        if importance_cols:
            # Scale importance values to [-1, 1] range
            for col in importance_cols:
                max_abs = abs(stats_df[col]).max()
                if max_abs > 0:
                    stats_df[f"{col}_scaled"] = stats_df[col] / max_abs
            
            scaled_cols = [f"{col}_scaled" for col in importance_cols]
            colors = ['#3498db', '#e67e22', '#9b59b6']  # Blue, orange, purple
            
            stats_df.plot(x='season', y=scaled_cols, kind='bar', ax=axes[1],
                         color=colors)
            axes[1].set_title('Relative Feature Importance by Season')
            axes[1].set_ylabel('Relative Importance (scaled)')
            axes[1].grid(True, alpha=0.3)
            
            # Fix legend labels
            handles = axes[1].get_legend().get_lines()
            labels = [col.replace('_importance_scaled', '').title() 
                     for col in scaled_cols]
            axes[1].legend(handles, labels, bbox_to_anchor=(1.05, 1),
                         loc='upper left')
            
            # Rotate labels
            axes[1].tick_params(axis='x', rotation=45)
        else:
            axes[1].text(0.5, 0.5, 'No feature importance data available',
                        horizontalalignment='center',
                        verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'seasonal_patterns.png')
    except Exception as e:
        print(f"Warning: Error creating plots: {str(e)}")
    finally:
        plt.close()

def main():
    """Analyze seasonal patterns in the data."""
    data_dir = Path('data/cdec')
    output_dir = data_dir / 'seasonal_analysis'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    data = load_data(data_dir)
    
    print("\nAnalyzing flow distributions...")
    flow_stats = analyze_flow_ranges(data, output_dir)
    print("\nFlow statistics by season:")
    print(flow_stats.to_string(index=False))
    
    print("\nAnalyzing feature transitions...")
    analyze_feature_transitions(data, output_dir)
    
    print("\nAnalyzing correlations...")
    analyze_correlations(data, output_dir)
    
    print("\nAnalyzing feature importance...")
    analyze_feature_importance(data, output_dir)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
