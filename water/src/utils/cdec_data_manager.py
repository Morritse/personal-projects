"""
CDEC Data Manager

Provides a flexible data structure for managing heterogeneous CDEC station data.
Key features:
- Handles missing data and different temporal resolutions
- Supports metadata about data availability
- Makes it easy to query across stations and sensors
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass
class SensorMetadata:
    """Metadata about a sensor's data availability."""
    sensor_code: str
    sensor_name: str
    durations: List[str]  # e.g., ['daily', 'hourly']
    units: str
    earliest_date: Optional[datetime] = None
    latest_date: Optional[datetime] = None
    data_quality: Optional[float] = None  # e.g., percentage of non-null values

@dataclass
class StationMetadata:
    """Metadata about a station."""
    station_id: str
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    elevation: Optional[float] = None
    sensors: Dict[str, SensorMetadata] = None

class CDECDataManager:
    """Manages CDEC station data and metadata."""
    
    def __init__(self, base_dir: str = "data/cdec"):
        """Initialize the data manager.
        
        Args:
            base_dir: Base directory for storing data and metadata
        """
        self.base_dir = Path(base_dir)
        self.stations_dir = self.base_dir / "stations"
        self.metadata_file = self.base_dir / "metadata.json"
        self.data_cache = {}  # In-memory cache for frequently accessed data
        
        # Create directory structure
        self.stations_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file or create new if doesn't exist."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'stations': {}, 'last_updated': None}
    
    def save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
    
    def add_station_data(
        self,
        station_id: str,
        sensor_code: str,
        duration: str,
        data: pd.DataFrame,
        metadata: Optional[Dict] = None
    ):
        """Add or update data for a station/sensor combination.
        
        Args:
            station_id: Station identifier
            sensor_code: Sensor code (e.g., '45' for precipitation)
            duration: Data duration ('daily', 'hourly', etc.)
            data: DataFrame with timestamp index and data columns
            metadata: Optional metadata about the sensor/data
        """
        # Create station directory if needed
        station_dir = self.stations_dir / station_id
        station_dir.mkdir(exist_ok=True)
        
        # Save data to parquet file (efficient storage format)
        filename = f"{sensor_code}_{duration}.parquet"
        data.to_parquet(station_dir / filename)
        
        # Update metadata
        if station_id not in self.metadata['stations']:
            self.metadata['stations'][station_id] = {
                'sensors': {},
                'last_updated': datetime.now().isoformat()
            }
        
        sensor_meta = self.metadata['stations'][station_id]['sensors']
        if sensor_code not in sensor_meta:
            sensor_meta[sensor_code] = {}
        
        sensor_meta[sensor_code].update({
            'duration': duration,
            'earliest_date': data.index.min().isoformat(),
            'latest_date': data.index.max().isoformat(),
            'columns': list(data.columns),
            **(metadata or {})
        })
        
        self.save_metadata()
    
    def get_station_data(
        self,
        station_id: str,
        sensor_code: str,
        duration: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Retrieve data for a station/sensor combination.
        
        Args:
            station_id: Station identifier
            sensor_code: Sensor code
            duration: Data duration
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            DataFrame with requested data
        """
        filename = self.stations_dir / station_id / f"{sensor_code}_{duration}.parquet"
        if not filename.exists():
            return pd.DataFrame()  # Return empty DataFrame if no data
            
        # Load data
        df = pd.read_parquet(filename)
        
        # Apply date filters if provided
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        return df
    
    def get_multi_station_data(
        self,
        stations: List[str],
        sensor_code: str,
        duration: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interpolate: bool = False
    ) -> pd.DataFrame:
        """Retrieve and combine data from multiple stations.
        
        Args:
            stations: List of station IDs
            sensor_code: Sensor code
            duration: Data duration
            start_date: Optional start date filter
            end_date: Optional end date filter
            interpolate: Whether to interpolate missing values
            
        Returns:
            DataFrame with combined data from all stations
        """
        dfs = []
        for station in stations:
            df = self.get_station_data(station, sensor_code, duration, start_date, end_date)
            if not df.empty:
                # Add station identifier to column names
                df.columns = [f"{station}_{col}" for col in df.columns]
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
            
        # Combine all DataFrames
        result = pd.concat(dfs, axis=1)
        
        if interpolate:
            result = result.interpolate(method='time')
            
        return result
    
    def get_available_sensors(self, station_id: str) -> Dict:
        """Get available sensors and their metadata for a station."""
        return self.metadata['stations'].get(station_id, {}).get('sensors', {})
    
    def get_common_sensors(self, stations: List[str], duration: Optional[str] = None) -> List[str]:
        """Find sensors that are common across multiple stations.
        
        Args:
            stations: List of station IDs
            duration: Optional duration filter
            
        Returns:
            List of sensor codes available across all stations
        """
        common_sensors = None
        for station in stations:
            sensors = set(self.get_available_sensors(station).keys())
            if duration:
                # Filter by duration if specified
                sensors = {
                    code for code in sensors 
                    if self.metadata['stations'][station]['sensors'][code]['duration'] == duration
                }
            
            if common_sensors is None:
                common_sensors = sensors
            else:
                common_sensors &= sensors
                
        return sorted(list(common_sensors or set()))

def example_usage():
    """Example of how to use the CDECDataManager."""
    manager = CDECDataManager()
    
    # Example: Add some data
    data = pd.DataFrame({
        'value': [1.2, 1.5, 1.8],
        'quality': ['good', 'good', 'suspect']
    }, index=pd.date_range('2023-01-01', periods=3, freq='D'))
    
    manager.add_station_data(
        'SHA',
        '45',  # precipitation
        'daily',
        data,
        metadata={'units': 'inches', 'sensor_name': 'Precipitation'}
    )
    
    # Example: Query data
    sha_precip = manager.get_station_data('SHA', '45', 'daily')
    
    # Example: Get data from multiple stations
    multi_station = manager.get_multi_station_data(
        ['SHA', 'ORO', 'FOL'],
        '45',
        'daily',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31)
    )
    
    # Example: Find common sensors
    common = manager.get_common_sensors(['SHA', 'ORO', 'FOL'], duration='daily')
    
    return sha_precip, multi_station, common

if __name__ == "__main__":
    example_usage()
