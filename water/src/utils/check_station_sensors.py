"""
Script to check available sensors for CDEC stations.
"""

import requests
from datetime import datetime, timedelta
from typing import List, Dict, Set, Tuple

def get_station_sensors(station_id: str) -> Set[Tuple[str, str, str]]:
    """Query available sensors for a station using a recent time window."""
    
    base_url = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
    
    # Use last 7 days as sample window
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Format dates as MM/DD/YYYY
    start_date_fmt = start_date.strftime('%m/%d/%Y')
    end_date_fmt = end_date.strftime('%m/%d/%Y')
    
    # Test all sensor types and durations
    sensor_codes = [
        ('3', 'SNOW, WATER CONTENT', ['daily', 'event', 'hourly', 'monthly']),
        ('4', 'TEMPERATURE, AIR', ['hourly']),
        ('9', 'WIND, SPEED', ['event']),
        ('10', 'WIND, DIRECTION', ['event']),
        ('14', 'BATTERY VOLTAGE', ['event', 'hourly']),
        ('18', 'SNOW DEPTH', ['daily', 'event', 'hourly', 'monthly']),
        ('30', 'TEMPERATURE, AIR AVERAGE', ['daily']),
        ('31', 'TEMPERATURE, AIR MAXIMUM', ['daily']),
        ('32', 'TEMPERATURE, AIR MINIMUM', ['daily']),
        ('82', 'SNOW, WATER CONTENT(REVISED)', ['daily', 'monthly']),
        ('45', 'PRECIPITATION', ['hourly', 'daily']),
        ('25', 'TEMPERATURE', ['hourly', 'daily']),
        ('1', 'STAGE', ['hourly', 'daily']),
        ('20', 'FLOW', ['hourly', 'daily']),
        ('15', 'STORAGE', ['hourly', 'daily']),
        ('6', 'EVAPORATION', ['daily']),
        ('8', 'FULL NATURAL FLOW', ['daily', 'monthly'])
    ]
    
    available_sensors = set()
    
    print(f"\nChecking sensors for station {station_id}...")
    
    for sensor_code, sensor_name, durations in sensor_codes:
        for duration in durations:
            # Map duration to CDEC duration code
            if duration == 'daily':
                dur_code = 'D'
            elif duration == 'hourly':
                dur_code = 'H'
            elif duration == 'event':
                dur_code = 'E'
            else:  # monthly
                dur_code = 'M'
                
            params = {
                'Stations': station_id,
                'SensorNums': sensor_code,
                'dur_code': dur_code,
                'Start': start_date_fmt,
                'End': end_date_fmt
            }
            
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                
                # If we get data beyond the header, the sensor is available
                lines = [line for line in response.text.strip().split('\n') if line.strip()]
                if len(lines) > 1:  # More than just header
                    available_sensors.add((sensor_code, duration, sensor_name))
                    print(f"✓ {sensor_code}-({duration}) - {sensor_name}")
                
            except requests.exceptions.RequestException as e:
                print(f"Error checking sensor {sensor_code} ({duration}): {e}")
                continue
            
    return available_sensors

def main():
    """Check sensors for a list of stations."""
    stations = [
        'AGP', 'AGW', 'BDM', 'CNV', 'FLR', 'FRT', 'GRM', 'GTM', 'HNT',
        'KKR', 'KSP', 'MIL', 'NFR', 'NTP', 'PSR', 'RDN', 'SBF', 'TMR', 'VLC'
    ]
    
    station_sensors = {}
    
    for station in stations:
        sensors = get_station_sensors(station)
        station_sensors[station] = sensors
        
    # Print summary
    print("\nSensor Summary:")
    print("=" * 80)
    for station, sensors in station_sensors.items():
        print(f"\n{station}:")
        if sensors:
            # Group by sensor type
            sensor_groups = {}
            for code, duration, name in sorted(sensors):
                if (code, name) not in sensor_groups:
                    sensor_groups[(code, name)] = []
                sensor_groups[(code, name)].append(duration)
                
            for (code, name), durations in sorted(sensor_groups.items()):
                durations_str = ", ".join(sorted(durations))
                print(f"  - {code} - {name} ({durations_str})")
        else:
            print("  No active sensors found")

if __name__ == "__main__":
    main()
