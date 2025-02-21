"""
Compare single model vs seasonal models to demonstrate trade-offs.

This script trains:
1. A single model for all seasons
2. Separate seasonal models
3. Compares their performance and feature usage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Tuple

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

def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    """Prepare features including seasonal indicators."""
    feature_dfs = []
    
    # Raw features
    feature_dfs.append(X)
    
    # Recent changes
    for lag in [1, 2, 3]:
        # Lags
        lagged = X.shift(lag)
        lagged.columns = [f"{col}_lag_{lag}d" for col in X.columns]
        feature_dfs.append(lagged)
        
        # Deltas
        delta = X - X.shift(lag)
        delta.columns = [f"{col}_delta_{lag}d" for col in X.columns]
        feature_dfs.append(delta)
    
    # Add seasonal features
    seasonal = pd.DataFrame(index=X.index)
    # Annual cycle
    seasonal['annual_sin'] = np.sin(2 * np.pi * X.index.dayofyear / 365.25)
    seasonal['annual_cos'] = np.cos(2 * np.pi * X.index.dayofyear / 365.25)
    # Semi-annual cycle
    seasonal['semiannual_sin'] = np.sin(4 * np.pi * X.index.dayofyear / 365.25)
    seasonal['semiannual_cos'] = np.cos(4 * np.pi * X.index.dayofyear / 365.25)
    # Month indicators
    for month in range(1, 13):
        seasonal[f'month_{month}'] = (X.index.month == month).astype(int)
    feature_dfs.append(seasonal)
    
    return pd.concat(feature_dfs, axis=1)

def train_single_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: int = 30
) -> Tuple[GradientBoostingRegressor, Dict]:
    """Train a single model on all data."""
    # Prepare features
    X_full = prepare_features(X)
    
    # Split train/test
    train_idx = X_full.index < X_full.index.max() - pd.Timedelta(days=test_size)
    X_train = X_full[train_idx]
    X_test = X_full[~train_idx]
    y_train = y[train_idx]
    y_test = y[~train_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.7,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate overall
    y_pred = model.predict(X_test_scaled)
    overall_r2 = r2_score(y_test, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Evaluate by season
    seasons = pd.Series(y_test.index.map(get_season))
    seasonal_metrics = {}
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        mask = seasons == season
        if mask.any():
            seasonal_metrics[season] = {
                'r2': r2_score(y_test[mask], y_pred[mask]),
                'rmse': np.sqrt(mean_squared_error(y_test[mask], y_pred[mask])),
                'mean_flow': y_test[mask].mean()
            }
    
    return model, {
        'overall_r2': overall_r2,
        'overall_rmse': overall_rmse,
        'seasonal_metrics': seasonal_metrics,
        'feature_importance': pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
    }

def train_seasonal_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: int = 30
) -> Tuple[Dict[str, GradientBoostingRegressor], Dict]:
    """Train separate models for each season."""
    # Prepare features
    X_full = prepare_features(X)
    
    # Get seasons
    seasons = pd.Series(X_full.index.map(get_season))
    
    # Train models
    models = {}
    metrics = {}
    feature_importance = {}
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        # Get seasonal data
        mask = seasons == season
        X_season = X_full[mask]
        y_season = y[mask]
        
        # Split train/test
        train_idx = X_season.index < X_season.index.max() - pd.Timedelta(days=test_size)
        X_train = X_season[train_idx]
        X_test = X_season[~train_idx]
        y_train = y_season[train_idx]
        y_test = y_season[~train_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.01,
            subsample=0.7,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Store model
        models[season] = model
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics[season] = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mean_flow': y_test.mean()
        }
        
        # Store feature importance
        feature_importance[season] = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)
    
    return models, {
        'metrics': metrics,
        'feature_importance': feature_importance
    }

def compare_approaches(single_results: Dict, seasonal_results: Dict, output_dir: Path):
    """Compare and visualize the differences between approaches."""
    # Performance comparison
    performance = []
    
    # Single model performance by season
    for season, metrics in single_results['seasonal_metrics'].items():
        performance.append({
            'season': season,
            'model': 'Single',
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mean_flow': metrics['mean_flow']
        })
    
    # Seasonal models performance
    for season, metrics in seasonal_results['metrics'].items():
        performance.append({
            'season': season,
            'model': 'Seasonal',
            'r2': metrics['r2'],
            'rmse': metrics['rmse'],
            'mean_flow': metrics['mean_flow']
        })
    
    # Create comparison plots
    performance_df = pd.DataFrame(performance)
    
    # R² comparison
    plt.figure(figsize=(12, 6))
    performance_df.pivot(index='season', columns='model', values='r2').plot(kind='bar')
    plt.title('R² Score Comparison')
    plt.ylabel('R²')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / 'r2_comparison.png')
    plt.close()
    
    # Feature importance comparison
    plt.figure(figsize=(15, 10))
    
    # Get top features from single model
    single_top = single_results['feature_importance'].head(10)
    
    # Get seasonal top features
    seasonal_top = {}
    for season, importance in seasonal_results['feature_importance'].items():
        seasonal_top[season] = importance.head(5)
    
    # Plot
    n_seasons = len(seasonal_top)
    fig, axes = plt.subplots(n_seasons + 1, 1, figsize=(15, 4*(n_seasons + 1)))
    
    # Single model
    single_top.plot(kind='barh', ax=axes[0])
    axes[0].set_title('Single Model - Top 10 Features')
    
    # Seasonal models
    for i, (season, importance) in enumerate(seasonal_top.items(), 1):
        importance.plot(kind='barh', ax=axes[i])
        axes[i].set_title(f'{season.title()} Model - Top 5 Features')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_comparison.png')
    plt.close()

def main():
    """Compare modeling approaches."""
    # Setup
    data_dir = Path('data/cdec')
    output_dir = data_dir / 'model_comparison'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    X = pd.read_csv(data_dir / 'sensor_data.csv', parse_dates=['DATE TIME'], index_col='DATE TIME')
    y = pd.read_csv(data_dir / 'flow_data.csv', parse_dates=['DATE TIME'], index_col='DATE TIME')['flow']
    
    print("Training single model...")
    single_model, single_results = train_single_model(X, y)
    
    print("\nTraining seasonal models...")
    seasonal_models, seasonal_results = train_seasonal_models(X, y)
    
    print("\nComparing approaches...")
    compare_approaches(single_results, seasonal_results, output_dir)
    
    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    main()
