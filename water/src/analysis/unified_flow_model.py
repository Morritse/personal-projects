"""
Train a unified flow model that incorporates seasonal patterns.

Key improvements:
1. Uses all data but adds seasonal context
2. More sophisticated feature engineering
3. Better scale handling
4. Incorporates hidden patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
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

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data."""
    # Load flow data
    print("\nLoading flow data...")
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
    
    sensor_data = []
    for category, types in sensor_types.items():
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
                    df = df.rename(columns={station_id: f"{sensor_type}_{station_id}"})
                    sensor_data.append(df)
                    print(f"Loaded station: {station_id}")
    
    # Combine all data
    print("\nCombining features...")
    X = pd.concat(sensor_data, axis=1)
    print(f"Initial features: {X.shape[1]}")
    
    # Align dates and handle missing values
    common_dates = X.index.intersection(flow.index)
    X = X.loc[common_dates]
    y = flow.loc[common_dates]
    
    # Handle missing values
    X = X.ffill(limit=7).bfill(limit=7)
    X = X.dropna(axis=1, thresh=len(X)*0.7)
    X = X.fillna(X.mean())
    
    print(f"Final features: {X.shape[1]}")
    return X, y

def engineer_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Create sophisticated features incorporating seasonal patterns."""
    feature_dfs = []
    
    # Group features by type
    snow_cols = [col for col in X.columns if 'snow' in col.lower()]
    temp_cols = [col for col in X.columns if 'temperature' in col.lower()]
    precip_cols = [col for col in X.columns if 'precipitation' in col.lower()]
    storage_cols = [col for col in X.columns if 'storage' in col.lower()]
    
    print(f"\nFeature groups: {len(snow_cols)} snow, {len(temp_cols)} temp, "
          f"{len(precip_cols)} precip, {len(storage_cols)} storage")
    
    # 1. Recent changes with different windows per feature type
    for cols, windows, name in [
        (snow_cols, [1, 2, 3, 7, 14, 30], 'snow'),  # Snow changes slowly
        (temp_cols, [1, 2, 3, 7], 'temp'),          # Temperature changes quickly
        (precip_cols, [1, 2, 3, 7, 14], 'precip'),  # Mix of timescales
        (storage_cols, [1, 2, 3, 7], 'storage')     # Storage changes moderately
    ]:
        if not cols:
            continue
            
        X_subset = X[cols]
        
        # Raw lags with forward fill
        print(f"\nCreating lag features for {name}...")
        for lag in windows:
            # Simple lag features
            lagged = X_subset.shift(lag)
            lagged = lagged.fillna(method='ffill')
            lagged.columns = [f"{col}_lag_{lag}d" for col in cols]
            feature_dfs.append(lagged)
            
            # Rolling statistics with proper handling
            print(f"Creating rolling features for {lag} day window...")
            
            # Mean (require at least half the window)
            min_periods = max(1, lag // 2)
            rolled_mean = X_subset.rolling(window=lag, min_periods=min_periods).mean()
            rolled_mean = rolled_mean.fillna(method='ffill').fillna(method='bfill')
            rolled_mean.columns = [f"{col}_mean_{lag}d" for col in cols]
            feature_dfs.append(rolled_mean)
            
            # Standard deviation (require at least 2 values)
            if lag >= 2:
                rolled_std = X_subset.rolling(window=lag, min_periods=2).std()
                rolled_std = rolled_std.fillna(method='ffill').fillna(method='bfill')
                rolled_std.columns = [f"{col}_std_{lag}d" for col in cols]
                feature_dfs.append(rolled_std)
            
            # Min/Max range (require at least 2 values)
            if lag >= 2:
                rolled_min = X_subset.rolling(window=lag, min_periods=2).min()
                rolled_max = X_subset.rolling(window=lag, min_periods=2).max()
                rolled_range = (rolled_max - rolled_min).fillna(0)
                rolled_range.columns = [f"{col}_range_{lag}d" for col in cols]
                feature_dfs.append(rolled_range)
    
    # 2. Seasonal context
    seasonal = pd.DataFrame(index=X.index)
    
    # Basic cycles
    seasonal['day_of_year'] = X.index.dayofyear / 365.25
    seasonal['annual_sin'] = np.sin(2 * np.pi * seasonal['day_of_year'])
    seasonal['annual_cos'] = np.cos(2 * np.pi * seasonal['day_of_year'])
    seasonal['semiannual_sin'] = np.sin(4 * np.pi * seasonal['day_of_year'])
    seasonal['semiannual_cos'] = np.cos(4 * np.pi * seasonal['day_of_year'])
    
    # Season indicators
    seasons = pd.Series(X.index.map(get_season), index=X.index)
    for season in ['winter', 'spring', 'summer', 'fall']:
        seasonal[f'is_{season}'] = (seasons == season).astype(float)
    
    # Month indicators
    for month in range(1, 13):
        seasonal[f'month_{month}'] = (X.index.month == month).astype(float)
    
    feature_dfs.append(seasonal)
    
    # 3. Interaction features
    # Snow-temperature interactions with NaN handling
    if snow_cols and temp_cols:
        snow_mean = X[snow_cols].mean(axis=1).fillna(0)  # 0 if no snow data
        temp_mean = X[temp_cols].mean(axis=1).fillna(X[temp_cols].mean().mean())  # Mean temp if missing
        interactions = pd.DataFrame(index=X.index)
        interactions['snow_temp'] = snow_mean * temp_mean
        # Add individual means for context
        interactions['snow_mean'] = snow_mean
        interactions['temp_mean'] = temp_mean
        feature_dfs.append(interactions)
    
    # 4. Target-derived features
    # Previous day's flow with careful handling
    if isinstance(y, pd.Series):
        flow_features = pd.DataFrame(index=X.index)
        
        # Fill any gaps in the flow data first
        y_filled = y.copy()
        y_filled = y_filled.fillna(method='ffill', limit=3)  # Forward fill up to 3 days
        y_filled = y_filled.fillna(y_filled.mean())  # Use mean for longer gaps
        
        for lag in [1, 2, 3, 7, 14, 30]:
            lagged_flow = y_filled.shift(lag)
            flow_features[f'flow_lag_{lag}d'] = lagged_flow
            
            # Add rolling means for each lag
            flow_features[f'flow_mean_{lag}d'] = y_filled.rolling(lag, min_periods=1).mean()
        
        feature_dfs.append(flow_features)
    
    # Combine and clean features
    print("\nCombining features...")
    X_full = pd.concat(feature_dfs, axis=1)
    print(f"Combined shape: {X_full.shape}")
    
    # Check for NaNs before cleaning
    nan_cols = X_full.isna().sum()
    if nan_cols.any():
        print("\nNaN counts before cleaning:")
        print(nan_cols[nan_cols > 0])
    
    # Handle any remaining NaNs carefully
    print("\nCleaning features...")
    for col in X_full.columns:
        if X_full[col].isna().any():
            if 'lag' in col:
                # Use forward fill for lags
                X_full[col] = X_full[col].fillna(method='ffill')
            elif 'mean' in col:
                # Use column mean for averages
                X_full[col] = X_full[col].fillna(X_full[col].mean())
            elif 'std' in col or 'range' in col:
                # Use 0 for missing variability measures
                X_full[col] = X_full[col].fillna(0)
            else:
                # Use column mean for everything else
                X_full[col] = X_full[col].fillna(X_full[col].mean())
    
    # Verify no NaNs remain
    nan_cols = X_full.isna().sum()
    if nan_cols.any():
        print("\nWarning: NaN values remain after cleaning:")
        print(nan_cols[nan_cols > 0])
        print("Filling remaining NaNs with 0")
        X_full = X_full.fillna(0)  # Last resort: fill with 0
    
    print(f"Engineered features: {X_full.shape[1]}")
    return X_full

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    test_size: int = 30
) -> Tuple[GradientBoostingRegressor, Dict]:
    """Train unified model with seasonal awareness."""
    # Prepare target
    y_target = y.shift(-1)  # Predict next day's flow
    
    # Remove samples with NaN targets
    valid_idx = ~y_target.isna()
    X_clean = X.loc[valid_idx]
    y_clean = y_target.loc[valid_idx]
    
    # Split train/test
    train_idx = X_clean.index < X_clean.index.max() - pd.Timedelta(days=test_size)
    X_train = X_clean[train_idx]
    X_test = X_clean[~train_idx]
    y_train = y_clean[train_idx]
    y_test = y_clean[~train_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle negative values in target
    if y_train.min() < 0:
        print("\nAdjusting for negative flows...")
        offset = abs(min(y_train.min(), y_test.min())) + 1
        y_train_adj = y_train + offset
        y_test_adj = y_test + offset
    else:
        y_train_adj = y_train
        y_test_adj = y_test
    
    # Log transform for better scale handling
    y_train_log = np.log1p(y_train_adj)
    y_test_log = np.log1p(y_test_adj)
    
    # Train model
    print("\nTraining model...")
    model = GradientBoostingRegressor(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.01,
        subsample=0.7,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=42
    )
    model.fit(X_train_scaled, y_train_log)
    
    # Predict and evaluate
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)
    if y_train.min() < 0:
        y_pred = y_pred - offset
    
    # Calculate metrics
    metrics = {
        'r2_log': r2_score(y_test_log, y_pred_log),
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred)
    }
    
    print("\nModel Performance:")
    print(f"  Log-scale R²: {metrics['r2_log']:.3f}")
    print(f"  Raw-scale R²: {metrics['r2']:.3f}")
    print(f"  RMSE: {metrics['rmse']:.1f}")
    print(f"  MAE: {metrics['mae']:.1f}")
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
    
    # Predictions
    ax1.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
    ax1.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
    ax1.set_title('Model Predictions')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Flow')
    ax1.legend()
    ax1.grid(True)
    
    # Residuals
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals vs Predicted')
    ax2.set_xlabel('Predicted Flow')
    ax2.set_ylabel('Residual')
    ax2.grid(True)
    
    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=True)
    importance.tail(20).plot(kind='barh', ax=ax3)
    ax3.set_title('Top 20 Important Features')
    ax3.set_xlabel('Importance')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_analysis.png')
    plt.close()
    
    # Print top features
    print("\nTop 10 important features:")
    for feat, imp in importance.tail(10)[::-1].items():
        print(f"  {feat}: {imp:.3f}")
    
    return model, metrics

def main():
    """Train unified flow model."""
    data_dir = Path('data/cdec')
    output_dir = data_dir / 'unified_model'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    X, y = load_data(data_dir)
    
    # Engineer features
    X_full = engineer_features(X, y)
    
    # Train model
    model, metrics = train_model(X_full, y, output_dir)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
