"""
Simplified flow model focusing on key features and proper scaling.

Key improvements:
1. Fewer, more meaningful features
2. Better handling of negative flows
3. Separate scaling per season
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare key features."""
    # Load flow data
    print("\nLoading flow data...")
    flow_dir = data_dir / 'full_natural_flow'
    flow_df = pd.read_csv(flow_dir / 'SBF.csv',
                         parse_dates=['DATE TIME'],
                         index_col='DATE TIME')
    flow = pd.to_numeric(flow_df['SBF'], errors='coerce')
    
    # Create base features
    data = pd.DataFrame(index=flow.index)
    
    # Flow history (carefully handled)
    for lag in [1, 2, 3, 7]:
        col = f'flow_lag_{lag}d'
        data[col] = flow.shift(lag)
        # Forward fill up to the lag amount
        data[col] = data[col].fillna(method='ffill', limit=lag)
    
    # Rolling statistics (with proper window sizes)
    for window in [3, 7, 14]:
        # Mean flow
        data[f'flow_mean_{window}d'] = (
            flow.rolling(window, min_periods=1)
            .mean()
            .fillna(method='ffill', limit=window)
        )
        # Flow variability
        data[f'flow_std_{window}d'] = (
            flow.rolling(window, min_periods=2)
            .std()
            .fillna(0)  # No variation if not enough data
        )
    
    # Seasonal context
    data['day_of_year'] = data.index.dayofyear / 365.25
    data['annual_sin'] = np.sin(2 * np.pi * data['day_of_year'])
    data['annual_cos'] = np.cos(2 * np.pi * data['day_of_year'])
    
    # Season indicators
    data['season'] = data.index.map(get_season)
    for season in ['winter', 'spring', 'summer', 'fall']:
        data[f'is_{season}'] = (data['season'] == season).astype(float)
    
    # Drop any remaining NaN values
    data = data.dropna()
    flow = flow[data.index]
    
    print(f"Final features: {data.shape[1]}")
    return data, flow

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    test_size: int = 30
) -> Tuple[GradientBoostingRegressor, Dict]:
    """Train model with careful handling of scales."""
    # Get season for each sample
    seasons = X['season']
    
    # Remove season column and indicators (used differently)
    X = X.drop(['season'] + [col for col in X.columns if col.startswith('is_')], axis=1)
    
    # Split train/test
    train_idx = X.index < X.index.max() - pd.Timedelta(days=test_size)
    X_train = X[train_idx]
    X_test = X[~train_idx]
    y_train = y[train_idx]
    y_test = y[~train_idx]
    seasons_train = seasons[train_idx]
    seasons_test = seasons[~train_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle negative values by season
    y_train_adj = pd.Series(index=y_train.index, dtype=float)
    y_test_adj = pd.Series(index=y_test.index, dtype=float)
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        # Get seasonal mask
        train_mask = seasons_train == season
        test_mask = seasons_test == season
        
        if train_mask.any():
            # Calculate offset if needed
            season_min = min(
                y_train[train_mask].min(),
                y_test[test_mask].min() if test_mask.any() else float('inf')
            )
            if season_min < 0:
                offset = abs(season_min) + 1
            else:
                offset = 0
            
            # Apply offset and log transform
            y_train_adj[train_mask] = np.log1p(y_train[train_mask] + offset)
            if test_mask.any():
                y_test_adj[test_mask] = np.log1p(y_test[test_mask] + offset)
    
    # Train model
    print("\nTraining model...")
    model = GradientBoostingRegressor(
        n_estimators=500,  # Fewer trees
        max_depth=5,       # Simpler trees
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train_scaled, y_train_adj)
    
    # Predict and evaluate by season
    metrics = {}
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        # Get seasonal mask
        test_mask = seasons_test == season
        if not test_mask.any():
            continue
        
        # Get predictions
        y_pred_log = model.predict(X_test_scaled[test_mask])
        
        # Transform back if needed
        season_min = min(
            y_train[seasons_train == season].min(),
            y_test[test_mask].min()
        )
        if season_min < 0:
            offset = abs(season_min) + 1
            y_pred = np.expm1(y_pred_log) - offset
        else:
            y_pred = np.expm1(y_pred_log)
        
        # Calculate metrics
        metrics[season] = {
            'r2': r2_score(y_test[test_mask], y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test[test_mask], y_pred)),
            'mae': mean_absolute_error(y_test[test_mask], y_pred)
        }
    
    # Print results
    print("\nPerformance by Season:")
    for season, m in metrics.items():
        print(f"\n{season.title()}:")
        print(f"  R²: {m['r2']:.3f}")
        print(f"  RMSE: {m['rmse']:.1f}")
        print(f"  MAE: {m['mae']:.1f}")
    
    # Plot feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=True)
    
    plt.figure(figsize=(10, 6))
    importance.plot(kind='barh')
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png')
    plt.close()
    
    print("\nTop 10 important features:")
    for feat, imp in importance.tail(10)[::-1].items():
        print(f"  {feat}: {imp:.3f}")
    
    return model, metrics

def main():
    """Train simplified flow model."""
    data_dir = Path('data/cdec')
    output_dir = data_dir / 'simple_model'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    X, y = load_data(data_dir)
    
    # Train model
    model, metrics = train_model(X, y, output_dir)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
