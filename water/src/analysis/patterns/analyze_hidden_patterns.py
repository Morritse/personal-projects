"""
Analyze hidden patterns in sensor data using unsupervised learning.

This script:
1. Uses PCA to find important feature combinations
2. Applies clustering to find natural groupings
3. Visualizes temporal and spatial patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple

def load_sensor_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all sensor data from organized directories.
    
    Args:
        data_dir: Base directory containing sensor data
        
    Returns:
        Dictionary mapping sensor types to their data
    """
    sensor_data = {}
    
    # Load each sensor type
    for sensor_dir in data_dir.iterdir():
        if not sensor_dir.is_dir():
            continue
            
        sensor_type = sensor_dir.name
        print(f"\nLoading {sensor_type} data...")
        
        # Load all station files
        station_data = []
        for station_file in sensor_dir.glob('*.csv'):
            df = pd.read_csv(station_file,
                           parse_dates=['DATE TIME'],
                           index_col='DATE TIME')
            df = df.rename(columns={'VALUE': station_file.stem})
            station_data.append(df)
        
        if station_data:
            # Combine stations
            combined = pd.concat(station_data, axis=1)
            sensor_data[sensor_type] = combined
            print(f"Loaded {len(combined.columns)} stations")
    
    return sensor_data

def prepare_feature_matrix(sensor_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Prepare feature matrix from all sensor data.
    
    Args:
        sensor_data: Dictionary of sensor DataFrames
        
    Returns:
        Combined feature matrix
    """
    # Align all data on common dates
    common_dates = None
    for df in sensor_data.values():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)
    
    # Combine features
    feature_dfs = []
    for sensor_type, df in sensor_data.items():
        # Get data for common dates
        aligned = df.loc[common_dates]
        # Add sensor type prefix
        aligned.columns = [f"{sensor_type}_{col}" for col in aligned.columns]
        feature_dfs.append(aligned)
    
    # Combine all features
    X = pd.concat(feature_dfs, axis=1)
    
    # Handle missing values
    X = X.ffill(limit=7).bfill(limit=7)  # Fill gaps up to 7 days
    X = X.dropna(axis=1, thresh=len(X)*0.7)  # Keep columns with at least 70% data
    X = X.fillna(X.mean())
    
    return X

def analyze_pca(X: pd.DataFrame, output_dir: Path, n_components: int = 10) -> Tuple[PCA, np.ndarray]:
    """Perform PCA analysis.
    
    Args:
        X: Feature matrix
        output_dir: Directory to save plots
        n_components: Number of components to keep
        
    Returns:
        Fitted PCA model and transformed data
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Plot explained variance
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_explained_variance.png')
    plt.close()
    
    # Plot feature importance
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=X.columns
    )
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(components_df.head(20), cmap='RdBu', center=0)
    plt.title('Top 20 Features in Principal Components')
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_components.png')
    plt.close()
    
    return pca, X_pca

def analyze_clusters(X_pca: np.ndarray, dates: pd.DatetimeIndex, output_dir: Path, n_clusters: int = 5):
    """Perform clustering analysis.
    
    Args:
        X_pca: PCA-transformed data
        dates: Dates corresponding to samples
        output_dir: Directory to save plots
        n_clusters: Number of clusters
    """
    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    
    # Plot clusters over time
    plt.figure(figsize=(15, 5))
    plt.scatter(dates, [0]*len(dates), c=clusters, cmap='Set3', alpha=0.6)
    plt.title('Clusters Over Time')
    plt.xlabel('Date')
    plt.yticks([])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'clusters_timeline.png')
    plt.close()
    
    # Plot seasonal distribution
    cluster_months = pd.DataFrame({
        'month': dates.month,
        'cluster': clusters
    })
    
    plt.figure(figsize=(12, 6))
    for cluster in range(n_clusters):
        cluster_data = cluster_months[cluster_months['cluster'] == cluster]
        plt.hist(cluster_data['month'], bins=12, alpha=0.3, label=f'Cluster {cluster}')
    plt.xlabel('Month')
    plt.ylabel('Count')
    plt.title('Cluster Distribution Across Months')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_seasonality.png')
    plt.close()

def main():
    """Analyze hidden patterns in the data."""
    # Create output directory
    output_dir = Path('data/cdec/patterns')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving analysis outputs to {output_dir}")
    
    # Load all sensor data
    sensor_data = load_sensor_data(Path('data/cdec'))
    
    # Prepare feature matrix
    X = prepare_feature_matrix(sensor_data)
    print(f"\nPrepared feature matrix: {X.shape[1]} features, {len(X)} samples")
    
    # Perform PCA
    print("\nPerforming PCA analysis...")
    pca, X_pca = analyze_pca(X, output_dir)
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.cumsum()[:5]}")
    
    # Perform clustering
    print("\nPerforming cluster analysis...")
    analyze_clusters(X_pca, X.index, output_dir)

if __name__ == "__main__":
    main()
