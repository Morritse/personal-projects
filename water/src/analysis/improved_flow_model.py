"""
Improved flow model combining:
1. Original build_flow_model.py strengths
2. Seasonal awareness
3. Better negative flow handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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

def get_feature_lags(horizon: int, feature_type: str) -> List[int]:
    """Get appropriate feature lags based on type and horizon."""
    # Base lags for all features
    lags = [1, 2, 3, 7, 14]
    
    # Add longer lags for longer horizons
    if horizon > 7:
        lags.extend([30, 45, 60])
    
    # Add specific lags based on feature type
    if 'snow' in feature_type.lower():
        # Snow changes slowly
        lags.extend([30, 60, 90])
    elif 'temperature' in feature_type.lower():
        # Temperature has daily patterns
        lags.extend([1, 2, 3, 4, 5])
    elif 'precipitation' in feature_type.lower():
        # Precipitation has delayed effects
        lags.extend([5, 10, 15, 20])
    elif 'storage' in feature_type.lower():
        # Storage changes slowly
        lags.extend([30, 45, 60])
    
    return sorted(list(set(lags)))

def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data with careful handling."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
    # Load flow data
    print("\nLoading flow data...")
    flow_dir = data_dir / 'full_natural_flow'
    flow_file = flow_dir / 'SBF.csv'
    
    if not flow_file.exists():
        raise FileNotFoundError(f"Flow data file not found: {flow_file}")
        
    try:
        flow_df = pd.read_csv(flow_file,
                            parse_dates=['DATE TIME'],
                            index_col='DATE TIME')
        flow = pd.to_numeric(flow_df['SBF'], errors='coerce')
    except Exception as e:
        raise RuntimeError(f"Error loading flow data: {str(e)}")
    
    # Load sensor data
    sensor_data = []
    sensor_types = ['snow_water_content', 'snow_depth', 'precipitation',
                   'temperature_air_average', 'storage']
    
    for sensor_type in sensor_types:
        sensor_dir = data_dir / sensor_type
        if not sensor_dir.exists():
            print(f"Warning: Sensor directory not found: {sensor_dir}")
            continue
            
        print(f"\nLoading {sensor_type} data...")
        station_files = list(sensor_dir.glob('*.csv'))
        
        if not station_files:
            print(f"Warning: No CSV files found in {sensor_dir}")
            continue
            
        for station_file in station_files:
            try:
                df = pd.read_csv(station_file,
                               parse_dates=['DATE TIME'],
                               index_col='DATE TIME')
                station_id = station_file.stem
                df[f"{sensor_type}_{station_id}"] = pd.to_numeric(df[station_id], errors='coerce')
                if not df.empty:
                    sensor_data.append(df)
                    print(f"Loaded station: {station_id}")
                else:
                    print(f"Warning: Empty data for station: {station_id}")
            except Exception as e:
                print(f"Error loading {station_file}: {str(e)}")
                continue
    
    # Combine all data
    print("\nCombining features...")
    X = pd.concat(sensor_data, axis=1)
    
    # Align dates
    common_dates = X.index.intersection(flow.index)
    X = X.loc[common_dates]
    y = flow.loc[common_dates]
    
    print(f"Final dataset: {X.shape[1]} features, {len(X)} samples")
    return X, y

def engineer_features(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int = 1
) -> pd.DataFrame:
    """Create features with careful handling of each type."""
    feature_dfs = []
    
    # Group features by type
    feature_groups = {
        'snow': [col for col in X.columns if 'snow' in col.lower()],
        'temperature': [col for col in X.columns if 'temperature' in col.lower()],
        'precipitation': [col for col in X.columns if 'precipitation' in col.lower()],
        'storage': [col for col in X.columns if 'storage' in col.lower()]
    }
    
    # Process each feature group
    for feature_type, cols in feature_groups.items():
        if not cols:
            continue
            
        print(f"\nProcessing {feature_type} features...")
        X_subset = X[cols]
        
        # Get appropriate lags
        lags = get_feature_lags(horizon, feature_type)
        
        for lag in lags:
            # Raw lags
            lagged = X_subset.shift(lag)
            lagged = lagged.fillna(method='ffill', limit=lag)
            lagged.columns = [f"{col}_lag_{lag}d" for col in cols]
            feature_dfs.append(lagged)
            
            # Rolling statistics
            if lag >= 3:  # Only for longer windows
                # Mean
                rolled_mean = (
                    X_subset.rolling(lag, min_periods=max(2, lag//3))
                    .mean()
                    .fillna(method='ffill', limit=lag)
                )
                rolled_mean.columns = [f"{col}_mean_{lag}d" for col in cols]
                feature_dfs.append(rolled_mean)
                
                # Standard deviation
                rolled_std = (
                    X_subset.rolling(lag, min_periods=2)
                    .std()
                    .fillna(0)
                )
                rolled_std.columns = [f"{col}_std_{lag}d" for col in cols]
                feature_dfs.append(rolled_std)
    
    # Add flow history
    flow_lags = [1, 2, 3, 7, 14, 30][:max(3, horizon)]
    for lag in flow_lags:
        # Previous flows
        lagged_flow = pd.DataFrame(y.shift(lag), columns=[f'flow_lag_{lag}d'])
        feature_dfs.append(lagged_flow)
        
        # Rolling statistics
        if lag >= 3:
            # Mean flow
            mean_flow = pd.DataFrame(
                y.rolling(lag, min_periods=max(2, lag//3)).mean(),
                columns=[f'flow_mean_{lag}d']
            )
            feature_dfs.append(mean_flow)
            
            # Flow variability
            std_flow = pd.DataFrame(
                y.rolling(lag, min_periods=2).std(),
                columns=[f'flow_std_{lag}d']
            )
            feature_dfs.append(std_flow)
    
    # Add seasonal context
    seasonal = pd.DataFrame(index=X.index)
    
    # Continuous seasonal signal
    seasonal['day_of_year'] = seasonal.index.dayofyear / 365.25
    seasonal['annual_sin'] = np.sin(2 * np.pi * seasonal['day_of_year'])
    seasonal['annual_cos'] = np.cos(2 * np.pi * seasonal['day_of_year'])
    
    # Season indicators
    seasonal['season'] = seasonal.index.map(get_season)
    for season in ['winter', 'spring', 'summer', 'fall']:
        seasonal[f'is_{season}'] = (seasonal['season'] == season).astype(float)
    
    feature_dfs.append(seasonal.drop(['season', 'day_of_year'], axis=1))
    
    # Combine all features
    X_full = pd.concat(feature_dfs, axis=1)
    
    # Handle missing values
    X_full = X_full.ffill(limit=horizon).bfill(limit=horizon)
    
    # Drop features with too many missing values
    missing_ratio = X_full.isna().mean()
    good_cols = missing_ratio[missing_ratio < 0.3].index
    X_full = X_full[good_cols]
    
    # Fill remaining NaNs with 0
    X_full = X_full.fillna(0)
    
    print(f"Engineered features: {X_full.shape[1]}")
    return X_full

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    horizon: int = 1,
    n_splits: int = 5,
    output_dir: Path = None
) -> Tuple[GradientBoostingRegressor, Dict]:
    """Train model with proper validation and seasonal handling."""
    # Initialize cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Track performance by season
    seasonal_metrics = {
        'winter': [], 'spring': [], 'summer': [], 'fall': []
    }
    
    # Get seasons for each sample
    seasons = pd.Series(X.index.map(get_season), index=X.index)
    
    # Initialize model
    model = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42
    )
    
    print("\nTraining model...")
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        # Split data
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate by season
        test_seasons = seasons.iloc[test_idx]
        y_pred = model.predict(X_test_scaled)
        
        for season in seasonal_metrics.keys():
            season_mask = test_seasons == season
            if season_mask.any():
                metrics = {
                    'r2': r2_score(y_test[season_mask], y_pred[season_mask]),
                    'rmse': np.sqrt(mean_squared_error(y_test[season_mask], y_pred[season_mask])),
                    'mae': mean_absolute_error(y_test[season_mask], y_pred[season_mask])
                }
                seasonal_metrics[season].append(metrics)
        
        # Plot fold predictions
        if output_dir:
            plt.figure(figsize=(15, 5))
            plt.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
            plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
            plt.title(f'Fold {fold + 1} Predictions')
            plt.xlabel('Date')
            plt.ylabel('Flow')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / f'fold_{fold+1}_predictions.png')
            plt.close()
    
    # Print average performance by season
    print("\nAverage Performance by Season:")
    for season, metrics_list in seasonal_metrics.items():
        if metrics_list:
            avg_metrics = {
                metric: np.mean([m[metric] for m in metrics_list])
                for metric in ['r2', 'rmse', 'mae']
            }
            print(f"\n{season.title()}:")
            print(f"  R²: {avg_metrics['r2']:.3f}")
            print(f"  RMSE: {avg_metrics['rmse']:.1f}")
            print(f"  MAE: {avg_metrics['mae']:.1f}")
    
    # Plot feature importance
    if output_dir:
        importance = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=True)
        
        plt.figure(figsize=(12, 8))
        importance.tail(20).plot(kind='barh')
        plt.title('Top 20 Important Features')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(output_dir / 'feature_importance.png')
        plt.close()
        
        print("\nTop 10 important features:")
        for feat, imp in importance.tail(10)[::-1].items():
            print(f"  {feat}: {imp:.3f}")
    
    return model, seasonal_metrics

def main():
    """Train improved flow model."""
    try:
        # Get project root directory (2 levels up from script location)
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / 'data/cdec'
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        output_dir = data_dir / 'improved_model'
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data
        X, y = load_data(data_dir)
        
        if X.empty or y.empty:
            raise ValueError("No valid data loaded")
            
        # Train models for different horizons
        for horizon in [1, 7, 30]:
            print(f"\nTraining {horizon}-day ahead model...")
            horizon_dir = output_dir / f'{horizon}d_ahead'
            horizon_dir.mkdir(exist_ok=True)
            
            # Engineer features
            X_full = engineer_features(X, y, horizon=horizon)
            
            # Prepare target
            y_target = y.shift(-horizon)
            
            # Align features and target
            common_idx = X_full.index.intersection(y_target.dropna().index)
            X_full = X_full.loc[common_idx]
            y_target = y_target.loc[common_idx]
            
            # Train model
            model, metrics = train_model(
                X_full,
                y_target,
                horizon=horizon,
                output_dir=horizon_dir
            )
        
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
