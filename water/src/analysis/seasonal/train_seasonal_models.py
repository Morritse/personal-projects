"""
Train separate models for each season to better capture seasonal patterns.

Seasonal models offer several advantages over a single model:
1. Each season has distinct patterns and relationships:
   - Winter: Snow accumulation and rainfall dominate
   - Spring: Snowmelt and temperature are key drivers
   - Summer: Temperature and evaporation have stronger effects
   - Fall: Early precipitation and base flow patterns

2. Feature importance varies by season:
   - Snow features matter more in winter/spring
   - Temperature more important in summer
   - Precipitation patterns differ seasonally

3. Flow ranges and variability differ significantly:
   - Spring typically has highest flows (snowmelt)
   - Summer often has lowest flows
   - Different error distributions and scales

4. By training separate models, each can specialize in:
   - Season-specific feature relationships
   - Different flow ranges and variability
   - Seasonal error patterns
   - Optimal feature combinations

This script:
1. Splits data into seasonal groups
2. Trains specialized models for each season
3. Analyzes seasonal performance differences
4. Compares feature importance across seasons
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
    """Get season for a given date.
    
    Args:
        date: Date to get season for
        
    Returns:
        Season name (winter, spring, summer, fall)
    """
    month = date.month
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    else:
        return 'fall'

def load_and_prepare_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for seasonal modeling.
    
    Args:
        data_dir: Directory containing sensor data
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Load flow data from station files
    print("\nLoading flow data...")
    flow_data = []
    flow_dir = data_dir / 'full_natural_flow'
    
    if not flow_dir.exists():
        raise ValueError(f"Flow data directory not found: {flow_dir}")
    
    for flow_file in flow_dir.glob('*.csv'):
        try:
            # Read data
            df = pd.read_csv(flow_file,
                           parse_dates=['DATE TIME'],
                           index_col='DATE TIME')
            
            # Get station ID from filename
            station_id = flow_file.stem
            if station_id not in df.columns:
                print(f"Warning: No flow data found in {flow_file.name}")
                print(f"Available columns: {df.columns.tolist()}")
                continue
            
            # Convert to numeric
            df[station_id] = pd.to_numeric(df[station_id], errors='coerce')
            df = df.dropna()
            
            if not df.empty:
                flow_data.append(df)
                print(f"Loaded flow station: {station_id}")
            else:
                print(f"Warning: No valid data in {flow_file.name}")
        except Exception as e:
            print(f"Error loading {flow_file.name}: {str(e)}")
            continue
    
    if not flow_data:
        raise ValueError("No valid flow data found")
    
    # Combine flow data
    flow = pd.concat(flow_data, axis=1)
    print(f"Flow data: {len(flow)} observations from {len(flow.columns)} stations")
    
    # Load sensor data
    sensor_data = []
    valid_sensor_types = {
        'precipitation', 'snow_water_content', 'snow_depth', 'storage',
        'temperature_air_average', 'temperature_air_maximum', 'temperature_air_minimum',
        'evaporation'
    }
    
    print("\nChecking directories:")
    for sensor_dir in data_dir.glob('*'):
        if not sensor_dir.is_dir():
            continue
            
        print(f"Found directory: {sensor_dir.name}")
        if sensor_dir.name not in valid_sensor_types:
            print(f"Skipping non-sensor directory: {sensor_dir.name}")
            continue
            
        sensor_type = sensor_dir.name
        print(f"\nLoading {sensor_type} data...")
        
        station_count = 0
        for station_file in sensor_dir.glob('*.csv'):
            try:
                # Read data
                df = pd.read_csv(station_file,
                               parse_dates=['DATE TIME'],
                               index_col='DATE TIME')
                
                # Station files have station ID as column name
                station_id = station_file.stem
                if station_id not in df.columns:
                    print(f"Warning: No data column found in {station_file.name}")
                    print(f"Available columns: {df.columns.tolist()}")
                    continue
                
                # Convert to numeric, handling bad values
                df[station_id] = pd.to_numeric(df[station_id], errors='coerce')
                df = df.dropna()
                
                if not df.empty:
                    df = df.rename(columns={station_id: f"{sensor_type}_{station_id}"})
                    sensor_data.append(df)
                    station_count += 1
                else:
                    print(f"Warning: No valid data in {station_file.name}")
            except Exception as e:
                print(f"Error loading {station_file.name}: {str(e)}")
                continue
        
        print(f"Loaded {station_count} stations")
    
    if not sensor_data:
        raise ValueError("No valid sensor data found")
    
    # Combine all features
    print("\nCombining features...")
    X = pd.concat(sensor_data, axis=1)
    print(f"Initial feature matrix: {X.shape[1]} features, {len(X)} samples")
    
    # Align with flow data
    common_dates = X.index.intersection(flow.index)
    X = X.loc[common_dates]
    y = flow.loc[common_dates]
    print(f"After alignment: {X.shape[1]} features, {len(X)} samples")
    
    # Handle missing values
    X = X.ffill(limit=7).bfill(limit=7)
    X = X.dropna(axis=1, thresh=len(X)*0.7)
    print(f"After removing sparse columns: {X.shape[1]} features")
    
    X = X.fillna(X.mean())
    print(f"Final feature matrix: {X.shape[1]} features, {len(X)} samples")
    
    return X, y

def train_seasonal_models(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    horizon: int = 1
) -> Dict[str, GradientBoostingRegressor]:
    """Train separate models for each season.
    
    Args:
        X: Feature matrix
        y: Target vector
        output_dir: Directory to save results
        horizon: Prediction horizon in days
        
    Returns:
        Dictionary mapping seasons to trained models
    """
    # Add season labels
    seasons = pd.Series(X.index.map(get_season), index=X.index)
    
    print("\nPreparing features...")
    feature_dfs = []
    
    # Group features by type for targeted processing
    snow_cols = [col for col in X.columns if 'snow' in col.lower()]
    temp_cols = [col for col in X.columns if 'temperature' in col.lower()]
    precip_cols = [col for col in X.columns if 'precipitation' in col.lower()]
    storage_cols = [col for col in X.columns if 'storage' in col.lower()]
    
    print(f"Found features: {len(snow_cols)} snow, {len(temp_cols)} temperature, "
          f"{len(precip_cols)} precipitation, {len(storage_cols)} storage")
    
    # Recent changes (deltas)
    for lag in [1, 2, 3]:
        # Raw lags
        lagged = X.shift(lag)
        lagged.columns = [f"{col}_lag_{lag}d" for col in X.columns]
        feature_dfs.append(lagged)
        
        # Changes
        delta = X - X.shift(lag)
        delta.columns = [f"{col}_delta_{lag}d" for col in X.columns]
        feature_dfs.append(delta)
    
    # Longer-term averages with different windows per feature type
    for cols, windows, name in [
        (snow_cols, [7, 14, 30, 90], 'snow'),  # Snow changes slowly
        (temp_cols, [3, 7, 14], 'temp'),       # Temperature changes quickly
        (precip_cols, [3, 7, 14, 30], 'precip'),  # Mix of fast and slow effects
        (storage_cols, [7, 14, 30], 'storage')  # Storage changes moderately
    ]:
        if not cols:
            continue
            
        X_subset = X[cols]
        for window in windows:
            # Rolling mean
            rolled_mean = X_subset.rolling(window=window, min_periods=1).mean()
            rolled_mean.columns = [f"{col}_avg_{window}d" for col in cols]
            feature_dfs.append(rolled_mean)
            
            # Rolling std to capture variability
            rolled_std = X_subset.rolling(window=window, min_periods=1).std()
            rolled_std.columns = [f"{col}_std_{window}d" for col in cols]
            feature_dfs.append(rolled_std)
    
    # Add seasonal features
    seasonal = pd.DataFrame(index=X.index)
    # Annual cycle
    seasonal['annual_sin'] = np.sin(2 * np.pi * X.index.dayofyear / 365.25)
    seasonal['annual_cos'] = np.cos(2 * np.pi * X.index.dayofyear / 365.25)
    # Semi-annual cycle
    seasonal['semiannual_sin'] = np.sin(4 * np.pi * X.index.dayofyear / 365.25)
    seasonal['semiannual_cos'] = np.cos(4 * np.pi * X.index.dayofyear / 365.25)
    # Month indicators for discrete seasonal effects
    for month in range(1, 13):
        seasonal[f'month_{month}'] = (X.index.month == month).astype(int)
    feature_dfs.append(seasonal)
    
    # Combine features
    X_full = pd.concat(feature_dfs, axis=1)
    
    # Prepare and clean target (use first flow station if multiple)
    if isinstance(y, pd.DataFrame):
        print(f"Multiple flow stations found: {y.columns.tolist()}")
        print(f"Using first station: {y.columns[0]}")
        y = y[y.columns[0]]
    
    y_target = y.shift(-horizon)
    
    # Drop any NaN values in target
    valid_idx = ~y_target.isna()
    n_dropped = (~valid_idx).sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} samples with NaN targets")
        X_full = X_full.loc[valid_idx]
        y_target = y_target.loc[valid_idx]
        seasons = seasons.loc[valid_idx]
    
    # Verify no NaN values remain
    if X_full.isna().any().any():
        print("Warning: NaN values in features - filling with mean")
        X_full = X_full.fillna(X_full.mean())
    
    if y_target.isna().any():
        raise ValueError("NaN values remain in target after cleaning")
    
    # Train models for each season
    models = {}
    metrics = []
    
    for season in ['winter', 'spring', 'summer', 'fall']:
        print(f"\nTraining {season} model...")
        
        # Get seasonal data
        mask = seasons == season
        X_season = X_full[mask]
        y_season = y_target[mask]
        print(f"Season {season}: {len(y_season)} samples")
        
        if len(y_season) == 0:
            print(f"Skipping {season} - no data")
            continue
        
        # Split into train/test
        train_idx = X_season.index < X_season.index.max() - pd.Timedelta(days=30)
        X_train = X_season[train_idx]
        X_test = X_season[~train_idx]
        y_train = y_season[train_idx]
        y_test = y_season[~train_idx]
        
        # Verify data is clean
        if X_train.isna().any().any() or X_test.isna().any().any():
            print("Warning: NaN values in split data - filling with training mean")
            train_means = X_train.mean()
            X_train = X_train.fillna(train_means)
            X_test = X_test.fillna(train_means)  # Use training mean for test data
        
        if y_train.isna().any() or y_test.isna().any():
            raise ValueError("NaN values in target after splitting")
        
        # Remove constant features
        std = X_train.std()
        good_cols = std > 0
        if not good_cols.all():
            print(f"Removing {(~good_cols).sum()} constant features")
            X_train = X_train.loc[:, good_cols]
            X_test = X_test.loc[:, good_cols]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check target range
        print(f"\nTarget range: {y_train.min():.1f} to {y_train.max():.1f}")
        
        # Handle negative values if present
        if y_train.min() < 0:
            print("Warning: Negative flow values present, offsetting...")
            offset = abs(min(y_train.min(), y_test.min())) + 1
            y_train_adj = y_train + offset
            y_test_adj = y_test + offset
        else:
            y_train_adj = y_train
            y_test_adj = y_test
        
        # Log transform target for better scale handling
        y_train_log = np.log1p(y_train_adj)
        y_test_log = np.log1p(y_test_adj)
        
        # Train model with tuned parameters
        model = GradientBoostingRegressor(
            n_estimators=1000,  # Many more trees for complex patterns
            max_depth=8,        # Deeper trees for feature interactions
            learning_rate=0.01,  # Much slower learning rate
            subsample=0.7,      # More aggressive subsampling
            min_samples_leaf=5, # Allow finer splits
            max_features=0.7,   # Feature subsampling
            random_state=42
        )
        model.fit(X_train_scaled, y_train_log)
        
        # Predict and transform back to original scale
        y_pred_log = model.predict(X_test_scaled)
        if y_train.min() < 0:
            y_pred = np.expm1(y_pred_log) - offset
        else:
            y_pred = np.expm1(y_pred_log)
        
        # Evaluate on both scales
        r2_log = r2_score(y_test_log, y_pred_log)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"  Log-scale R²: {r2_log:.3f}")
        print(f"  Raw-scale R²: {r2:.3f}")
        print(f"  RMSE: {rmse:.1f}")
        print(f"  MAE: {mae:.1f}")
        print(f"  Mean flow: {y_test.mean():.1f}")
        
        # Store results
        models[season] = model
        metrics.append({
            'season': season,
            'r2_log': r2_log,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mean_flow': y_test.mean(),
            'n_samples': len(y_season)
        })
        
        # Plot predictions, residuals, and feature importance
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Predictions plot with error bands
        residuals = y_test - y_pred
        std_resid = residuals.std()
        ax1.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
        ax1.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
        ax1.fill_between(y_test.index, 
                        y_pred - std_resid, 
                        y_pred + std_resid,
                        alpha=0.2, label='±1σ')
        ax1.set_title(f'{season.title()} Model Predictions')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Flow')
        ax1.legend()
        ax1.grid(True)
        
        # Residuals plot
        ax2.scatter(y_pred, residuals, alpha=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_title('Residuals vs Predicted')
        ax2.set_xlabel('Predicted Flow')
        ax2.set_ylabel('Residual')
        ax2.grid(True)
        
        # Feature importance plot
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=True)
        importance.tail(20).plot(kind='barh', ax=ax3)
        ax3.set_title('Top 20 Important Features')
        ax3.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{season}_analysis.png')
        plt.close()
        
        # Print feature importance summary
        print("\nTop 5 important features:")
        for feat, imp in importance.tail(5)[::-1].items():
            print(f"  {feat}: {imp:.3f}")
    
    # Plot performance comparison
    if metrics:
        metrics_df = pd.DataFrame(metrics)
        
        # Plot all metrics
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # R² scores
        metrics_df.plot(x='season', y=['r2', 'r2_log'], kind='bar', ax=ax1)
        ax1.set_title('R² Scores by Season')
        ax1.set_ylabel('R²')
        ax1.grid(True)
        
        # Error metrics
        metrics_df.plot(x='season', y=['rmse', 'mae'], kind='bar', ax=ax2)
        ax2.set_title('Error Metrics by Season')
        ax2.set_ylabel('Error')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'seasonal_performance.png')
        plt.close()
    else:
        print("\nNo models trained - no performance metrics to plot")
    
    return models

def main():
    """Train and evaluate seasonal models."""
    # Setup directories
    data_dir = Path('data/cdec')
    output_dir = data_dir / 'seasonal_models'
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    X, y = load_and_prepare_data(data_dir)
    
    print("\nTraining seasonal models...")
    models = train_seasonal_models(X, y, output_dir)
    
    print("\nResults saved to:", output_dir)

if __name__ == "__main__":
    main()
