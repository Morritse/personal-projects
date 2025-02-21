"""
Analyze historical flow data and its relationships with other sensor measurements.

This script:
1. Loads historical flow data
2. Organizes it into a pandas DataFrame with proper datetime indexing
3. Provides functions to analyze relationships between flow and other measurements
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

class HistoricalFlowAnalyzer:
    """Analyzes relationships between flow data and other measurements."""
    
    def __init__(self, data_dir: str = "data/cdec"):
        """Initialize the analyzer.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.flow_data = None
        self.sensor_data = {}
        
    def load_flow_data(self, flow_file: str):
        """Load historical flow data from CSV.
        
        Args:
            flow_file: Path to flow data CSV file
        """
        # Read the flow data
        df = pd.read_csv(flow_file)
        
        # Convert first column to datetime index
        df.index = pd.to_datetime(df.iloc[:, 0])
        df = df.iloc[:, 1:]  # Remove the date column since it's now the index
        
        # Convert to numeric, coercing errors to NaN
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Convert wide format (multiple years as columns) to long format (single time series)
        flow_data = []
        for col in df.columns:
            year_data = df[col].dropna()
            if not year_data.empty:
                flow_data.append(year_data)
                
        # Combine all non-empty series
        flow_series = pd.concat(flow_data)
        flow_series.sort_index(inplace=True)
        
        self.flow_data = flow_series
        print(f"Loaded flow data with {len(flow_series)} observations")
        print(f"Date range: {flow_series.index.min()} to {flow_series.index.max()}")
        
    def load_sensor_data(self, sensor_file: str, sensor_type: str):
        """Load sensor measurement data from CSV.
        
        Args:
            sensor_file: Path to sensor data CSV file
            sensor_type: Type of sensor (e.g., 'precipitation', 'snow_depth')
        """
        df = pd.read_csv(sensor_file)
        df.index = pd.to_datetime(df.iloc[:, 0])
        df = df.iloc[:, 1:]  # Remove date column
        
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        # Align index with flow data if available
        if self.flow_data is not None:
            common_dates = df.index.intersection(self.flow_data.index)
            df = df.loc[common_dates]
            
        self.sensor_data[sensor_type] = df
        print(f"Loaded {sensor_type} data with shape: {df.shape}")
        
    def analyze_flow_patterns(self, rolling_window: int = 7):
        """Analyze patterns in flow data.
        
        Args:
            rolling_window: Window size for rolling statistics
        """
        if self.flow_data is None:
            print("No flow data loaded")
            return
            
        # Calculate basic statistics
        stats = {
            'mean': self.flow_data.mean(),
            'std': self.flow_data.std(),
            'min': self.flow_data.min(),
            'max': self.flow_data.max()
        }
        
        # Calculate rolling statistics
        rolling_mean = self.flow_data.rolling(window=rolling_window, center=True).mean()
        rolling_std = self.flow_data.rolling(window=rolling_window, center=True).std()
        
        # Plot flow patterns
        plt.figure(figsize=(15, 10))
        
        # Plot time series with rolling mean
        plt.subplot(2, 1, 1)
        self.flow_data.plot(alpha=0.5, label='Daily Flow')
        rolling_mean.plot(label=f'{rolling_window}-day Rolling Mean')
        plt.title('Flow Time Series with Rolling Mean')
        plt.xlabel('Date')
        plt.ylabel('Flow')
        plt.legend()
        
        # Plot seasonal pattern
        plt.subplot(2, 1, 2)
        daily_means = self.flow_data.groupby(self.flow_data.index.dayofyear).mean()
        daily_std = self.flow_data.groupby(self.flow_data.index.dayofyear).std()
        daily_means.plot(label='Mean')
        plt.fill_between(
            daily_means.index,
            daily_means - daily_std,
            daily_means + daily_std,
            alpha=0.2,
            label='±1 std'
        )
        plt.title('Average Seasonal Pattern')
        plt.xlabel('Day of Year')
        plt.ylabel('Flow')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'flow_patterns.png')
        plt.close()
        
        return pd.Series(stats)
    
    def analyze_sensor_relationships(
        self,
        sensor_type: str,
        lag_days: List[int] = [0, 7, 14, 30],
        correlation_method: str = 'pearson',
        plot_relationships: bool = True
    ):
        """Analyze relationships between flow and sensor measurements.
        
        Args:
            sensor_type: Type of sensor to analyze
            lag_days: List of days to lag sensor data
            correlation_method: Method for correlation calculation
        """
        if sensor_type not in self.sensor_data:
            print(f"No {sensor_type} data loaded")
            return
            
        sensor_df = self.sensor_data[sensor_type]
        correlations = {}
        
        # Calculate correlations for each lag
        for lag in lag_days:
            lag_corr = {}
            for station in sensor_df.columns:
                # Get station data and align with flow
                station_data = sensor_df[station].dropna()
                if lag > 0:
                    station_data = station_data.shift(-lag)
                
                # Get overlapping dates
                common_dates = self.flow_data.index.intersection(station_data.index)
                if len(common_dates) > 30:  # Require at least 30 days of data
                    corr = self.flow_data[common_dates].corr(
                        station_data[common_dates],
                        method=correlation_method
                    )
                    lag_corr[station] = corr
                    
            correlations[f'lag_{lag}d'] = pd.Series(lag_corr)
            
        # Convert to DataFrame
        corr_df = pd.DataFrame(correlations)
        
        if plot_relationships:
            # Plot correlation heatmap
            plt.figure(figsize=(15, 10))
            
            # Correlation heatmap
            plt.subplot(2, 1, 1)
            sns.heatmap(
                corr_df,
                cmap='RdBu',
                center=0,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': f'Correlation ({correlation_method})'},
                xticklabels=True,
                yticklabels=True
            )
            plt.title(f'Flow vs {sensor_type} Correlations at Different Lags')
            plt.xlabel('Lag Time')
            plt.ylabel('Station')
            
            # Plot example relationship for station with highest correlation
            plt.subplot(2, 1, 2)
            best_station = corr_df['lag_0d'].abs().idxmax()
            station_data = sensor_df[best_station].dropna()
            common_dates = self.flow_data.index.intersection(station_data.index)
            
            # Normalize data for comparison
            flow_norm = (self.flow_data[common_dates] - self.flow_data[common_dates].mean()) / self.flow_data[common_dates].std()
            sensor_norm = (station_data[common_dates] - station_data[common_dates].mean()) / station_data[common_dates].std()
            
            flow_norm.plot(label='Flow (normalized)', alpha=0.7)
            sensor_norm.plot(label=f'{sensor_type} - {best_station} (normalized)', alpha=0.7)
            plt.title(f'Flow vs {sensor_type} Time Series (Best Correlation: {best_station})')
            plt.xlabel('Date')
            plt.ylabel('Normalized Value')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(self.data_dir / f'flow_{sensor_type}_analysis.png')
            plt.close()
        
        return corr_df
    
    def analyze_seasonal_patterns(self):
        """Analyze seasonal patterns in flow and sensor measurements."""
        if self.flow_data is None:
            print("No flow data loaded")
            return
            
        # Calculate seasonal statistics
        seasons = {
            'Winter': [12, 1, 2],
            'Spring': [3, 4, 5],
            'Summer': [6, 7, 8],
            'Fall': [9, 10, 11]
        }
        
        seasonal_stats = {}
        for season, months in seasons.items():
            mask = self.flow_data.index.month.isin(months)
            seasonal_stats[season] = {
                'mean': self.flow_data[mask].mean(),
                'std': self.flow_data[mask].std(),
                'max': self.flow_data[mask].max()
            }
            
        # Plot seasonal patterns
        plt.figure(figsize=(15, 10))
        
        # Box plot by month
        monthly_data = []
        labels = []
        for month in range(1, 13):
            mask = self.flow_data.index.month == month
            monthly_data.append(self.flow_data[mask].values.flatten())
            labels.append(pd.Timestamp(2000, month, 1).strftime('%b'))
            
        plt.boxplot(monthly_data, tick_labels=labels)
        plt.title('Monthly Flow Distribution')
        plt.ylabel('Flow')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'seasonal_patterns.png')
        plt.close()
        
        return pd.DataFrame(seasonal_stats)

def main():
    """Example usage of the HistoricalFlowAnalyzer."""
    analyzer = HistoricalFlowAnalyzer()
    
    # Load flow data
    analyzer.load_flow_data('data/cdec/comparative_sensor_8_full_natural_flow.csv')
    
    # Analyze flow patterns
    flow_stats = analyzer.analyze_flow_patterns()
    print("\nFlow Statistics:")
    print(flow_stats)
    print("\nFlow Statistics Description:")
    print("- Mean flow:", int(flow_stats['mean']), "units")
    print("- Standard deviation:", int(flow_stats['std']), "units")
    print("- Range:", int(flow_stats['min']), "to", int(flow_stats['max']), "units")
    
    # Analyze seasonal patterns
    seasonal_stats = analyzer.analyze_seasonal_patterns()
    print("\nSeasonal Statistics:")
    print(seasonal_stats)
    print("\nSeasonal Pattern Description:")
    for season in seasonal_stats.columns:
        print(f"\n{season}:")
        print(f"- Mean flow: {int(seasonal_stats[season]['mean'])} units")
        print(f"- Standard deviation: {int(seasonal_stats[season]['std'])} units")
        print(f"- Maximum flow: {int(seasonal_stats[season]['max'])} units")
    
    # Load and analyze relationships with other sensors
    sensor_files = {
        'precipitation': 'comparative_sensor_45_precipitation.csv',
        'snow': 'comparative_sensor_3_snow,_water_content.csv',
        'snow_revised': 'comparative_sensor_82_snow,_water_content(revised).csv',
        'storage': 'comparative_sensor_15_storage.csv'
    }
    
    print("\nSensor Relationship Analysis:")
    for sensor_type, filename in sensor_files.items():
        filepath = Path('data/cdec') / filename
        if filepath.exists():
            print(f"\n{sensor_type.upper()}:")
            analyzer.load_sensor_data(str(filepath), sensor_type)
            correlations = analyzer.analyze_sensor_relationships(
                sensor_type,
                lag_days=[0, 7, 14, 30, 60]  # Test different lags
            )
            
            # Find strongest correlations
            max_corr = correlations.abs().max().max()
            max_station = correlations.abs().max(axis=1).idxmax()
            max_lag = correlations.abs().max().idxmax()
            
            print(f"Strongest correlation: {max_corr:.3f}")
            print(f"Best station: {max_station}")
            print(f"Best lag: {max_lag}")
            print("\nCorrelation matrix:")
            print(correlations)

if __name__ == "__main__":
    main()
