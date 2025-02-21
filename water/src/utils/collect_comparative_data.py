"""
Collect comparative data across stations for flow analysis.

This script collects data for specific sensor types across all stations
and organizes it for comparison with full natural flow data from SBF station.
"""

import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from cdec_data_collector import CDECDataCollector

class ComparativeDataCollector:
    """Collects and organizes data for comparative analysis."""
    
    def __init__(self, base_dir: str = "data/cdec"):
        """Initialize the collector.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.collector = CDECDataCollector()
        self.stations_info = self._load_station_info()
        
    def _load_station_info(self) -> dict:
        """Load station sensor information."""
        with open(self.base_dir / "station_sensors.json", "r") as f:
            return json.load(f)
    
    def collect_sensor_data(
        self,
        sensor_code: str,
        start_date: str,
        end_date: str,
        output_prefix: str = ""
    ) -> pd.DataFrame:
        """Collect data for a specific sensor across all stations.
        
        Args:
            sensor_code: Sensor code to collect
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_prefix: Prefix for output files
            
        Returns:
            DataFrame with data from all stations for the sensor
        """
        print(f"\nCollecting data for sensor {sensor_code}")
        
        # Find stations with this sensor
        stations_with_sensor = []
        for station, info in self.stations_info["stations"].items():
            if sensor_code in info["sensors"]:
                if "daily" in info["sensors"][sensor_code]["durations"]:
                    stations_with_sensor.append(station)
        
        print(f"Found {len(stations_with_sensor)} stations with sensor {sensor_code}:")
        for station in stations_with_sensor:
            print(f"  - {station}")
        
        # Collect data for each station
        station_data = {}
        for station in stations_with_sensor:
            print(f"\nCollecting data for station {station}")
            raw_data = self.collector.fetch_station_data(
                station,
                sensor_code,
                duration_code="D",
                start_date=start_date,
                end_date=end_date
            )
            
            if raw_data:
                # Convert to DataFrame
                df = pd.DataFrame(raw_data)
                # Convert date and set as index
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df.set_index('date', inplace=True)
                
                # Handle missing or invalid values
                df['value'] = df['value'].replace('---', pd.NA)  # Replace missing values with pandas NA
                df['value'] = pd.to_numeric(df['value'], errors='coerce')  # Convert to numeric, invalid values become NA
                
                station_data[station] = df['value']
                print(f"  ✓ Collected {len(df)} records")
            else:
                print(f"  ✗ No data available")
        
        # Combine all station data
        if station_data:
            combined_df = pd.DataFrame(station_data)
            
            # Save to CSV
            sensor_name = self.get_sensor_name(sensor_code)
            filename = f"{output_prefix}sensor_{sensor_code}_{sensor_name.lower().replace(' ', '_')}.csv"
            filepath = self.base_dir / filename
            combined_df.to_csv(filepath)
            print(f"\nSaved combined data to {filepath}")
            
            return combined_df
        
        return pd.DataFrame()
    
    def get_sensor_name(self, sensor_code: str) -> str:
        """Get the name of a sensor from any station that has it."""
        for station_info in self.stations_info["stations"].values():
            if sensor_code in station_info["sensors"]:
                return station_info["sensors"][sensor_code]["name"]
        return "Unknown"
    
    def collect_flow_comparison_data(
        self,
        start_date: str,
        end_date: str,
        output_prefix: str = ""
    ):
        """Collect data for comparison with full natural flow.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_prefix: Prefix for output files
        """
        # Collect full natural flow data first
        print("\nCollecting full natural flow data from SBF")
        flow_data = self.collect_sensor_data("8", start_date, end_date, output_prefix)
        
        if flow_data.empty:
            print("No flow data available. Aborting collection.")
            return
        
        # Collect other relevant sensors
        sensor_codes = [
            "45",  # Precipitation
            "3",   # Snow Water Content
            "82",  # Snow Water Content (Revised)
            "15",  # Storage
        ]
        
        sensor_data = {}
        for sensor_code in sensor_codes:
            data = self.collect_sensor_data(sensor_code, start_date, end_date, output_prefix)
            if not data.empty:
                sensor_data[sensor_code] = data
        
        return {
            'flow': flow_data,
            'sensors': sensor_data
        }

def main():
    """Main function to collect comparative data."""
    collector = ComparativeDataCollector()
    
    # Start with a test period
    start_date = "2010-01-01"
    end_date = "2010-12-31"
    
    print(f"Collecting data from {start_date} to {end_date}")
    data = collector.collect_flow_comparison_data(
        start_date=start_date,
        end_date=end_date,
        output_prefix="comparative_"
    )
    
    if data and data['flow'] is not None:
        print("\nData collection complete!")
        print("\nAvailable data:")
        print("Flow data shape:", data['flow'].shape)
        for sensor_code, sensor_data in data['sensors'].items():
            name = collector.get_sensor_name(sensor_code)
            print(f"{name} data shape:", sensor_data.shape)

if __name__ == "__main__":
    main()
