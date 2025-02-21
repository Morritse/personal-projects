"""
Build predictive models for full natural flow using sensor data.

This script:
1. Prepares sensor data as features and flow data as target
2. Trains and evaluates different regression models
3. Analyzes feature importance and model performance
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, List, Tuple

class FlowPredictor:
    """Builds and evaluates models to predict full natural flow."""
    
    def __init__(self, data_dir: str = "data/cdec"):
        """Initialize the predictor.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.flow_data = None
        self.feature_data = {}
        self.models = {}
        self.scalers = {}
        
    def load_flow_data(self, flow_file: str = None):
        """Load flow data as the target variable."""
        if flow_file is None:
            flow_file = self.data_dir / "full_natural_flow/SBF.csv"
            
        if not Path(flow_file).exists():
            raise FileNotFoundError(f"Flow data file not found: {flow_file}")
            
        print("\nLoading flow data...")
        try:
            # Read CSV and fix column names
            df = pd.read_csv(flow_file)
            # Handle case where DATE TIME and SBF are combined
            if 'DATE TIMESBF' in df.columns:
                df.columns = ['datetime']
                # Split the datetime and flow columns
                df[['datetime', 'flow']] = df['datetime'].str.split('SBF', expand=True)
            else:
                df.columns = ['datetime', 'flow']
            
            # Convert datetime to index
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Convert flow to numeric
            self.flow_data = pd.to_numeric(df['flow'], errors='coerce')
            self.flow_data = self.flow_data.dropna()
            
        except Exception as e:
            raise RuntimeError(f"Error loading flow data: {str(e)}")
        
        print(f"Loaded flow data: {len(self.flow_data)} observations")
        print(f"Date range: {self.flow_data.index.min()} to {self.flow_data.index.max()}")
        print(f"Flow range: {self.flow_data.min():.1f} to {self.flow_data.max():.1f}")
        
    def load_sensor_data(self, sensor_file: str, sensor_type: str):
        """Load sensor data as features."""
        if not Path(sensor_file).exists():
            raise FileNotFoundError(f"Sensor data file not found: {sensor_file}")
            
        print(f"\nLoading {sensor_type} data...")
        try:
            # Read CSV and handle combined datetime/value format
            df = pd.read_csv(sensor_file)
            
            # Get station ID from filename
            station_id = Path(sensor_file).stem
            
            # Handle combined datetime/value format (e.g., 'DATE TIMEAGP')
            if f'DATE TIME{station_id}' in df.columns:
                df.columns = ['datetime']
                # Split datetime and value
                df[['datetime', 'value']] = df['datetime'].str.extract(f'(.+?)({station_id}.*)').values
                df['value'] = df['value'].str.replace(station_id, '')
            else:
                # Fallback to standard format
                df.columns = ['datetime', 'value']
            
            # Convert datetime and set as index
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Convert to numeric and handle missing values
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Align with flow data
            if self.flow_data is not None:
                common_dates = df.index.intersection(self.flow_data.index)
                df = df.loc[common_dates]
            
            # Add to feature data
            self.feature_data[sensor_type] = df
            print(f"Loaded {len(df.columns)} stations with {len(df)} observations")
            
        except Exception as e:
            raise RuntimeError(f"Error loading sensor data: {str(e)}")
        
    def prepare_data(
        self,
        target_shift: int = 1,
        feature_lags: List[int] = [0, 7, 14, 30],
        add_seasonal_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target vector y."""
        if self.flow_data is None:
            raise ValueError("Must load flow data first")
            
        # Prepare target (shifted flow)
        y = self.flow_data.shift(-target_shift).dropna()
        
        # Prepare features
        feature_dfs = []
        
        # Add lagged flow as feature
        for lag in feature_lags:
            lagged_flow = self.flow_data.shift(lag)
            lagged_flow.name = f'flow_lag_{lag}'
            feature_dfs.append(lagged_flow)
        
        # Add sensor features with lags if available
        if self.feature_data:
            for sensor_type, sensor_df in self.feature_data.items():
                for lag in feature_lags:
                    # Create lagged features with station ID in name
                    for col in sensor_df.columns:
                        lagged = sensor_df[col].shift(lag)
                        lagged.name = f"{sensor_type}_{lag}d"
                        feature_dfs.append(pd.DataFrame(lagged))
                        
                        # Add rolling means for longer lags
                        if lag >= 7:
                            rolled_mean = sensor_df[col].rolling(window=lag, min_periods=max(2, lag//3)).mean()
                            rolled_mean.name = f"{sensor_type}_mean_{lag}d"
                            feature_dfs.append(pd.DataFrame(rolled_mean))
        
        # Combine all features
        X = pd.concat(feature_dfs, axis=1)
        
        # Align features and target
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Add seasonal features
        if add_seasonal_features:
            # Ensure index is datetime
            if not isinstance(X.index, pd.DatetimeIndex):
                X.index = pd.to_datetime(X.index)
            
            # Create seasonal features all at once
            seasonal_features = pd.DataFrame(index=X.index)
            seasonal_features['day_sin'] = np.sin(2 * np.pi * seasonal_features.index.dayofyear / 365.25)
            seasonal_features['day_cos'] = np.cos(2 * np.pi * seasonal_features.index.dayofyear / 365.25)
            seasonal_features['month'] = seasonal_features.index.month
            seasonal_features['season'] = pd.cut(
                X.index.month,
                bins=[0, 3, 6, 9, 12],
                labels=['winter', 'spring', 'summer', 'fall']
            )
            
            # One-hot encode season
            season_dummies = pd.get_dummies(seasonal_features['season'], prefix='season')
            
            # Combine all features efficiently
            X = pd.concat([X, seasonal_features.drop('season', axis=1), season_dummies], axis=1)
        
        print(f"\nPreparing {X.shape[1]} features for {len(X)} samples")
        
        # Handle missing values more carefully
        print(f"\nData shape before handling missing values: {X.shape}")
        print(f"Missing values per column: {X.isna().sum().sum()}")
        
        # Forward fill with reasonable limits
        X = X.ffill(limit=7)
        
        # Drop columns with too many missing values
        missing_ratio = X.isna().mean()
        good_cols = missing_ratio[missing_ratio < 0.1].index
        X = X[good_cols]
        
        # Fill remaining gaps with column medians
        X = X.fillna(X.median())
        
        print(f"Data shape after handling missing values: {X.shape}")
        print(f"Remaining missing values: {X.isna().sum().sum()}")
        
        # Report data retention
        print(f"\nAfter handling missing values:")
        print(f"Final dataset shape: {X.shape}")
        
        if len(X) == 0:
            print("\nWARNING: No data samples remaining after processing!")
            print("Debug information:")
            print(f"- Flow data length: {len(self.flow_data)}")
            print(f"- Target data length: {len(y)}")
            for sensor_type, df in self.feature_data.items():
                print(f"- {sensor_type} data length: {len(df)}")
            return X, y
            
        # Plot data completeness over time
        plt.figure(figsize=(15, 5))
        completeness = (~X.isna()).astype(int).sum(axis=1) / X.shape[1]
        plt.plot(completeness.index, completeness.values, alpha=0.7)
        plt.title('Data Completeness Over Time')
        plt.xlabel('Date')
        plt.ylabel('Fraction of Features Available')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.data_dir / 'data_completeness.png')
        plt.close()
        
        # Print periods with most missing data
        print("\nPeriods with lowest data completeness:")
        worst_periods = completeness.nsmallest(5)
        for date, comp in worst_periods.items():
            print(f"  {date.strftime('%Y-%m-%d')}: {comp:.1%} complete")
        
        print(f"\nPrepared data with {X.shape[1]} features and {len(y)} samples")
        return X, y
    
    def train_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int = 1,
        n_splits: int = 5,
        random_state: int = 42
    ):
        """Train regression models."""
        # Initialize TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Initialize scaler
        scaler = StandardScaler()
        self.scalers['standard'] = scaler
        
        # Adjust model parameters based on prediction horizon
        if horizon <= 1:
            # Short-term: More specific models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=5,
                    random_state=random_state
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=random_state
                )
            }
        elif horizon <= 7:
            # Medium-term: More robust models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=10,
                    random_state=random_state
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=7,
                    learning_rate=0.05,
                    random_state=random_state
                )
            }
        else:
            # Long-term: More generalized models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_leaf=20,
                    random_state=random_state
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=300,
                    max_depth=10,
                    learning_rate=0.01,
                    subsample=0.8,
                    random_state=random_state
                )
            }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            fold_results = []
            
            # Cross-validation
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Handle problematic features before scaling
                X_train_std = X_train.std()
                bad_features = (X_train_std == 0) | X_train.isna().any()
                
                if bad_features.any():
                    n_constant = (X_train_std == 0).sum()
                    n_nan = X_train.isna().any().sum()
                    print(f"\nWarning: Removed {n_constant} constant and {n_nan} NaN features")
                    
                    # Keep only good features
                    X_train = X_train.loc[:, ~bad_features]
                    X_test = X_test.loc[:, ~bad_features]
                
                if X_train.empty or X_test.empty:
                    print("Error: No valid features remaining")
                    continue
                
                # Scale features
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred_test = model.predict(X_test_scaled)
                
                # Evaluate
                metrics = {
                    'r2': r2_score(y_test, y_pred_test),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    'mae': mean_absolute_error(y_test, y_pred_test)
                }
                fold_results.append(metrics)
                
                # Plot this fold's predictions
                plt.figure(figsize=(15, 5))
                plt.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
                plt.plot(y_test.index, y_pred_test, label='Predicted', alpha=0.7)
                plt.title(f'{name.replace("_", " ").title()} - Fold {fold+1} Predictions')
                plt.xlabel('Date')
                plt.ylabel('Flow')
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.data_dir / f'{name}_fold_{fold+1}_predictions.png')
                plt.close()
            
            # Store final model
            self.models[name] = model
            
            # Average results across folds
            results[name] = {
                metric: np.mean([fold[metric] for fold in fold_results])
                for metric in ['r2', 'rmse', 'mae']
            }
            
            # Plot predictions vs actual
            plt.figure(figsize=(15, 5))
            plt.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
            plt.plot(y_test.index, y_pred_test, label='Predicted', alpha=0.7)
            plt.title(f'{name.replace("_", " ").title()} - Test Set Predictions')
            plt.xlabel('Date')
            plt.ylabel('Flow')
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.data_dir / f'{name}_predictions.png')
            plt.close()
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                # Get feature importance for remaining features
                importance = pd.Series(
                    model.feature_importances_,
                    index=X_train.columns  # Use columns from filtered features
                ).sort_values(ascending=False)
                
                # Print top features
                print(f"\nTop 10 Important Features for {name.replace('_', ' ').title()}:")
                for feat, imp in importance.head(10).items():
                    print(f"  {feat}: {imp:.3f}")
                
                # Plot feature importance
                plt.figure(figsize=(15, 10))
                importance.head(20).plot(kind='bar')
                plt.title(f'{name.replace("_", " ").title()} - Top 20 Feature Importance')
                plt.xlabel('Feature')
                plt.ylabel('Importance')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(self.data_dir / f'{name}_h{horizon}_feature_importance.png')
                plt.close()
                
                # Store importance for later comparison
                if 'feature_importance' not in results[name]:
                    results[name]['feature_importance'] = importance
        
        return results

def plot_feature_importance_comparison(all_results, data_dir):
    """Plot feature importance comparison across horizons."""
    # Get all unique features
    all_features = set()
    for horizon_results in all_results.values():
        for model_results in horizon_results.values():
            if 'feature_importance' in model_results:
                all_features.update(model_results['feature_importance'].index)
    
    # Create comparison DataFrame
    comparison_data = []
    for horizon, horizon_results in all_results.items():
        for model_name, model_results in horizon_results.items():
            if 'feature_importance' in model_results:
                importance = model_results['feature_importance']
                comparison_data.append(pd.Series(
                    importance,
                    name=f'{model_name}_{horizon}day'
                ))
    
    comparison_df = pd.concat(comparison_data, axis=1).fillna(0)
    
    # Plot heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(comparison_df.head(15), annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Feature Importance Comparison Across Horizons')
    plt.tight_layout()
    plt.savefig(Path(data_dir) / 'feature_importance_comparison.png')
    plt.close()

def get_feature_lags(horizon: int, sensor_type: str) -> List[int]:
    """Get appropriate feature lags based on prediction horizon and sensor type."""
    # Base lags for all features
    lags = [1, 2, 3, 7, 14, 30]
    
    # Add longer lags for longer horizons
    if horizon > 7:
        lags.extend([45, 60, 90])
    
    # Add specific lags based on sensor type
    if 'snow' in sensor_type:
        # Snow changes more slowly, add longer lags
        lags.extend([120, 150, 180])
    elif 'temperature' in sensor_type:
        # Temperature has strong daily patterns
        lags.extend([1, 2, 3, 4, 5])
    elif 'precipitation' in sensor_type:
        # Precipitation can have delayed effects
        lags.extend([10, 20, 40])
    elif 'storage' in sensor_type:
        # Reservoir storage changes slowly
        lags.extend([90, 120, 150])
    
    # Remove duplicates and sort
    return sorted(list(set(lags)))

def check_water_year_alignment(data: pd.DataFrame) -> bool:
    """Check if data aligns with water year boundaries."""
    if data.empty:
        return True
        
    # Water years start on October 1st
    starts = data.index[data.index.month == 10]
    if len(starts) == 0:
        return True  # No October 1st dates to check
        
    for start in starts:
        if start.day != 1:
            return False
    return True

def main():
    """Train flow prediction models."""
    try:
        # Get project root directory (2 levels up from script location)
        project_root = Path(__file__).resolve().parents[2]
        data_dir = project_root / "data/cdec"
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
            
        predictor = FlowPredictor(str(data_dir))
        
        # Load flow data
        predictor.load_flow_data()
        
        # Load sensor data from individual station files
        sensor_dirs = {
            'snow': data_dir / 'snow_water_content',
            'precipitation': data_dir / 'precipitation',
            'storage': data_dir / 'storage'
        }
        
        # Load each station's data
        for sensor_type, sensor_dir in sensor_dirs.items():
            if not sensor_dir.exists():
                print(f"Warning: {sensor_type} directory not found: {sensor_dir}")
                continue
                
            station_files = list(sensor_dir.glob('*.csv'))
            if not station_files:
                print(f"Warning: No CSV files found in {sensor_dir}")
                continue
                
            for station_file in station_files:
                predictor.load_sensor_data(station_file, f"{sensor_type}_{station_file.stem}")
        
        # Train models with all sensor data
        print("\nTraining models with all sensor data...")
        horizons = [1, 7, 30]  # 1-day, 7-day, and 30-day ahead predictions
        
        # Store all results for comparison
        all_results = {}
        
        for horizon in horizons:
            print(f"\nTraining models for {horizon}-day ahead prediction:")
            
            # Get feature lags for each sensor type
            feature_lags = set()
            for sensor_type in predictor.feature_data.keys():
                lags = get_feature_lags(horizon, sensor_type)
                feature_lags.update(lags)
            
            # Sort lags for consistent order
            feature_lags = sorted(list(feature_lags))
            print(f"Using lags: {feature_lags}")
            
            X, y = predictor.prepare_data(
                target_shift=horizon,
                feature_lags=feature_lags,
                add_seasonal_features=True
            )
            
            # Train and evaluate models
            results = predictor.train_models(X, y, horizon=horizon, n_splits=5)
            
            # Store results for this horizon
            all_results[horizon] = results
            
            # Print results for this horizon
            print(f"\nModel Performance ({horizon}-day horizon):")
            for model_name, metrics in results.items():
                print(f"\n{model_name.replace('_', ' ').title()}:")
                print(f"  Average R² Score: {metrics['r2']:.3f}")
                print(f"  Average RMSE: {metrics['rmse']:.3f}")
                print(f"  Average MAE: {metrics['mae']:.3f}")
        
        # Plot feature importance comparison
        plot_feature_importance_comparison(all_results, predictor.data_dir)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main()
