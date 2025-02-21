"""
Collect and organize sensor data from CDEC.

This script:
1. Downloads data for key sensors based on prediction horizon
2. Organizes data into consistent format
3. Saves processed data for modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from datetime import datetime, timedelta
import time
import json
from io import StringIO

class CDECDataCollector:
    """Collects and processes CDEC sensor data."""
    
    def __init__(self, output_dir: str = "data/cdec"):
        """Initialize the collector.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Base URL for CDEC web services
        self.base_url = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
        
    def get_station_data(
        self,
        station_id: str,
        sensor_num: int,
        dur_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Get data for a specific station and sensor.
        
        Args:
            station_id: CDEC station ID
            sensor_num: Sensor number
            dur_code: Duration code (e.g., 'D' for daily)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with sensor data
        """
        params = {
            'Stations': station_id,
            'SensorNums': sensor_num,
            'dur_code': dur_code,
            'Start': start_date,
            'End': end_date
        }
        
        try:
            print(f"Requesting data for station {station_id} (sensor {sensor_num})...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            if not response.text.strip():
                print(f"No data returned for {station_id}")
                return pd.DataFrame()
            
            # Parse CSV data
            df = pd.read_csv(
                StringIO(response.text),
                parse_dates=['DATE TIME'],
                index_col='DATE TIME'
            )
            
            if df.empty:
                print(f"Empty dataset returned for {station_id}")
                return pd.DataFrame()
            
            # Clean up column names and values
            df.columns = [col.strip() for col in df.columns]
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Check for missing values
            missing = df['VALUE'].isna().sum()
            if missing > 0:
                print(f"Warning: {missing} missing values in {station_id} data")
            
            print(f"Retrieved {len(df)} observations for {station_id}")
            return df
            
        except requests.RequestException as e:
            print(f"Request error for {station_id}: {str(e)}")
            print(f"URL: {response.url if 'response' in locals() else 'unknown'}")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"Empty or malformed CSV from {station_id}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing data for {station_id}: {str(e)}")
            if 'response' in locals():
                print(f"Response content: {response.text[:200]}...")
            return pd.DataFrame()
    
    def collect_sensor_data(
        self,
        station_id: str,
        sensor_info: dict,
        start_date: str,
        end_date: str,
        sensor_dirs: dict
    ):
        """Collect data for all sensors at a station.
        
        Args:
            station_id: Station ID
            sensor_info: Dictionary of sensor numbers and their info
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            sensor_dirs: Dictionary mapping sensor types to output directories
        """
        # Group data by sensor type
        sensor_type_data = {}
        
        for sensor_num, info in sensor_info.items():
            sensor_type = info['name']
            print(f"\nCollecting {sensor_type} data from {station_id}...")
            
            df = self.get_station_data(
                station_id,
                sensor_num=int(sensor_num),
                dur_code='D',   # Daily
                start_date=start_date,
                end_date=end_date
            )
            
            if not df.empty:
                # Rename value column to include station ID
                df = df['VALUE'].rename(f"{station_id}")
                
                # Add to sensor type group
                if sensor_type not in sensor_type_data:
                    sensor_type_data[sensor_type] = []
                sensor_type_data[sensor_type].append(df)
            
            # Be nice to the server
            time.sleep(1)
        
        # Save data by sensor type
        for sensor_type, data_list in sensor_type_data.items():
            if data_list:
                # Combine all stations for this sensor type
                combined = pd.concat(data_list, axis=1)
                
                # Get output directory for this sensor type
                output_dir = sensor_dirs[sensor_type]
                
                # Save to file
                output_file = output_dir / f'{station_id}.csv'
                combined.to_csv(output_file)
                print(f"Saved {sensor_type} data from {station_id} to {output_file}")

def main():
    """Collect all available sensor data."""
    # Initialize collector
    collector = CDECDataCollector()
    
    # Set date range (last 10 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    
    # Load station sensor mappings
    with open('data/cdec/station_sensors.json', 'r') as f:
        station_sensors = json.load(f)
    
    # Print available sensor types
    print("\nAvailable sensor types:")
    for sensor_type in station_sensors['metadata']['sensor_types']:
        print(f"- {sensor_type}")
    
    # Create sensor type directories
    sensor_dirs = {}
    for sensor_type in station_sensors['metadata']['sensor_types']:
        dir_name = sensor_type.lower().replace(' ', '_').replace(',', '')
        sensor_dir = collector.output_dir / dir_name
        sensor_dir.mkdir(exist_ok=True)
        sensor_dirs[sensor_type] = sensor_dir
    
    # Group stations by sensor type
    sensor_type_stations = {}
    for station_id, info in station_sensors['stations'].items():
        for sensor_num, sensor_info in info['sensors'].items():
            sensor_type = sensor_info['name']
            if sensor_type not in sensor_type_stations:
                sensor_type_stations[sensor_type] = []
            sensor_type_stations[sensor_type].append(station_id)
    
    # Print summary of available data
    print("\nFound stations by sensor type:")
    for sensor_type, stations in sorted(sensor_type_stations.items()):
        print(f"- {sensor_type}: {len(stations)} stations")
    
    # Collect data from all stations
    print(f"\nCollecting data from {len(station_sensors['stations'])} stations...")
    for station_id, info in station_sensors['stations'].items():
        collector.collect_sensor_data(
            station_id,
            info['sensors'],
            start_date,
            end_date,
            sensor_dirs
        )

if __name__ == "__main__":
    main()
