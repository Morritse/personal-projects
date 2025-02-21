"""
Build CDEC Dataset

This script combines the CDEC data collector and manager to build a comprehensive dataset.
It will:
1. Check available sensors for each station
2. Download historical data
3. Store it in an organized data structure with metadata
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Set
from cdec_data_collector import CDECDataCollector
from cdec_data_manager import CDECDataManager

def build_dataset(
    stations: List[str],
    start_date: str,
    end_date: str,
    base_dir: str = "data/cdec"
):
    """Build a comprehensive dataset for the specified stations and date range.
    
    Args:
        stations: List of station IDs
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        base_dir: Base directory for storing data
    """
    collector = CDECDataCollector()
    manager = CDECDataManager(base_dir=base_dir)
    
    # Key sensor types we're interested in
    sensor_configs = [
        # (sensor_code, name, durations)
        ('3', 'SNOW, WATER CONTENT', ['daily', 'monthly']),
        ('18', 'SNOW DEPTH', ['daily', 'monthly']),
        ('82', 'SNOW, WATER CONTENT(REVISED)', ['daily', 'monthly']),
        ('30', 'TEMPERATURE, AIR AVERAGE', ['daily']),
        ('31', 'TEMPERATURE, AIR MAXIMUM', ['daily']),
        ('32', 'TEMPERATURE, AIR MINIMUM', ['daily']),
        ('45', 'PRECIPITATION', ['daily']),
        ('1', 'STAGE', ['daily']),
        ('20', 'FLOW', ['daily']),
        ('15', 'STORAGE', ['daily']),
        ('6', 'EVAPORATION', ['daily']),
        ('8', 'FULL NATURAL FLOW', ['daily', 'monthly'])
    ]
    
    for station in stations:
        print(f"\nProcessing station {station}...")
        
        for sensor_code, sensor_name, durations in sensor_configs:
            for duration in durations:
                print(f"  Collecting {sensor_name} ({duration}) data...")
                
                # Map duration to CDEC code
                dur_code = {
                    'daily': 'D',
                    'hourly': 'H',
                    'event': 'E',
                    'monthly': 'M'
                }[duration]
                
                # Collect data
                raw_data = collector.fetch_station_data(
                    station,
                    sensor_code,
                    duration_code=dur_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if raw_data:
                    # Convert to DataFrame
                    df = pd.DataFrame(raw_data)
                    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                    df.set_index('date', inplace=True)
                    
                    # Add to data manager
                    manager.add_station_data(
                        station,
                        sensor_code,
                        duration,
                        df,
                        metadata={
                            'sensor_name': sensor_name,
                            'units': df['units'].iloc[0] if 'units' in df else None
                        }
                    )
                    print(f"    ✓ Saved {len(df)} records")
                else:
                    print(f"    No data available")
    
    print("\nDataset build complete!")
    print("\nAvailable data summary:")
    for station in stations:
        print(f"\n{station}:")
        sensors = manager.get_available_sensors(station)
        for sensor_code, meta in sensors.items():
            print(f"  - {meta['sensor_name']} ({meta['duration']})")
            print(f"    Period: {meta['earliest_date']} to {meta['latest_date']}")
            print(f"    Units: {meta.get('units', 'unknown')}")

def main():
    """Main function to build the dataset."""
    # List of stations from your previous request
    stations = [
        'AGP', 'AGW', 'BDM', 'CNV', 'FLR', 'FRT', 'GRM', 'GTM', 'HNT',
        'KKR', 'KSP', 'MIL', 'NFR', 'NTP', 'PSR', 'RDN', 'SBF', 'TMR', 'VLC'
    ]
    
    # Start with a shorter time period for testing
    start_date = "2020-01-01"
    end_date = "2020-12-31"
    
    build_dataset(stations, start_date, end_date)

if __name__ == "__main__":
    main()
