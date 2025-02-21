"""
CDEC Data Collector

This script helps collect and organize historical weather station data from the California Data Exchange Center (CDEC).
Main functionalities:
- Fetch data from CDEC web services
- Process and validate the data
- Save in a format compatible with existing analysis
"""

import os
import json
import csv
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Optional, Union

class CDECDataCollector:
    """Handles data collection from California Data Exchange Center (CDEC)."""
    
    BASE_URL = "https://cdec.water.ca.gov/dynamicapp/req/CSVDataServlet"
    
    # Data codes from CDEC
    DATA_CODES = {
        'PRECIPITATION': '45',  # Hourly precipitation
        'TEMPERATURE': '25',    # Air temperature
        'SNOW_WATER': '82',     # Snow water content
        'STAGE': '1',          # River stage
        'FLOW': '20',          # River flow/discharge
        'STORAGE': '15'        # Reservoir storage
    }
    
    def __init__(self, output_dir: str = "data/cdec"):
        """Initialize the data collector.
        
        Args:
            output_dir: Directory to store collected data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def fetch_station_data(
        self,
        station_id: str,
        sensor_number: str,
        duration_code: str = "D",  # D for Daily
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """Fetch data for a specific station and sensor.
        
        Args:
            station_id: CDEC station identifier
            sensor_number: Type of measurement (see DATA_CODES)
            duration_code: Time interval (E:event, H:hourly, D:daily, M:monthly)
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            
        Returns:
            List of data points with timestamp and value
        """
        # Convert dates to MM/DD/YYYY format
        start_date_fmt = datetime.strptime(start_date, '%Y-%m-%d').strftime('%m/%d/%Y') if start_date else ''
        end_date_fmt = datetime.strptime(end_date, '%Y-%m-%d').strftime('%m/%d/%Y') if end_date else ''
        
        params = {
            'Stations': station_id,
            'SensorNums': sensor_number,
            'dur_code': duration_code,
            'Start': start_date_fmt,
            'End': end_date_fmt
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
            print(f"API URL: {response.url}")  # Print the actual URL being called
            
            # Parse CSV response
            if response.text.strip():
                # Split into lines and filter out empty ones
                lines = [line for line in response.text.strip().split('\n') if line.strip()]
                
                # Debug output
                print(f"Response first few lines:")
                for i, line in enumerate(lines[:5]):
                    print(f"Line {i}: {line}")
                
                if len(lines) > 1:  # Check if we have data beyond header
                    data = []
                    header = lines[0].split(',')  # Get header line
                    print(f"Header: {header}")
                    
                    # Find column indices from header
                    date_col = next((i for i, col in enumerate(header) if 'DATE' in col.upper()), 0)
                    value_col = next((i for i, col in enumerate(header) if 'VALUE' in col.upper()), 2)
                    units_col = next((i for i, col in enumerate(header) if 'UNITS' in col.upper()), -1)
                    
                    for line in lines[1:]:  # Skip header row
                        fields = [f.strip() for f in line.split(',')]
                        if len(fields) > max(date_col, value_col):  # Ensure we have required fields
                            try:
                                # Parse the date from YYYYMMDD format
                                date_str = fields[date_col].split()[0]  # Get just the date part, removing time
                                if len(date_str) == 8:  # YYYYMMDD format
                                    year = int(date_str[:4])
                                    month = int(date_str[4:6])
                                    day = int(date_str[6:8])
                                    date_obj = datetime(year, month, day)
                                else:
                                    # Try other formats if needed
                                    try:
                                        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
                                    except ValueError:
                                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                
                                data.append({
                                    'date': date_obj.strftime('%Y%m%d'),
                                    'value': fields[value_col].strip(),  # Remove any whitespace
                                    'units': fields[units_col] if units_col >= 0 and len(fields) > units_col else '',
                                    'stationId': station_id,
                                    'sensorType': sensor_number
                                })
                            except (ValueError, IndexError) as e:
                                print(f"Error processing line: {line}")
                                print(f"Error details: {e}")
                                continue
                    return data
                
            print(f"No data returned for station {station_id}")
            print(f"Response content: {response.text}")
            return []
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for station {station_id}: {e}")
            return []
            
    def process_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Process raw data from CDEC into standardized format.
        
        Args:
            raw_data: List of data points from CDEC
            
        Returns:
            Processed data in standardized format
        """
        processed_data = []
        
        for entry in raw_data:
            try:
                date = datetime.strptime(entry['date'], '%Y%m%d')
                value = float(entry['value']) if entry['value'] != '' else None
                
                processed_entry = {
                    'Date': date.strftime('%Y-%m-%d'),
                    'WaterYear': self._get_water_year(date),
                    'DayOfYear': date.timetuple().tm_yday,
                    'Month': date.month,
                    'Day': date.day,
                    'Value': value,
                    'Units': entry.get('units', ''),
                    'StationID': entry.get('stationId', ''),
                    'SensorType': entry.get('sensorType', '')
                }
                
                processed_data.append(processed_entry)
                
            except (ValueError, KeyError) as e:
                print(f"Error processing entry {entry}: {e}")
                continue
                
        return processed_data
    
    def _get_water_year(self, date: datetime) -> int:
        """Calculate water year for a given date.
        Water year starts October 1st of previous calendar year.
        """
        if date.month >= 10:
            return date.year + 1
        return date.year
    
    def save_to_csv(self, data: List[Dict], filename: str):
        """Save processed data to CSV file.
        
        Args:
            data: Processed data to save
            filename: Output filename
        """
        if not data:
            print("No data to save")
            return
            
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
                
            print(f"Data saved to {filepath}")
            
        except IOError as e:
            print(f"Error saving data to {filepath}: {e}")
    
    def collect_station_history(
        self,
        station_id: str,
        data_types: List[str],
        start_date: str,
        end_date: str,
        output_prefix: str = ""
    ):
        """Collect historical data for specified station and data types.
        
        Args:
            station_id: CDEC station identifier
            data_types: List of data types to collect (must match DATA_CODES keys)
            start_date: Start date in format 'YYYY-MM-DD'
            station_id: CDEC station identifier
            output_prefix: Prefix for output filenames
        """
        for data_type in data_types:
            if data_type not in self.DATA_CODES:
                print(f"Unknown data type: {data_type}")
                continue
                
            print(f"Collecting {data_type} data for station {station_id}")
            
            raw_data = self.fetch_station_data(
                station_id,
                self.DATA_CODES[data_type],
                start_date=start_date,
                end_date=end_date
            )
            
            if raw_data:
                processed_data = self.process_data(raw_data)
                filename = f"{output_prefix}{station_id}_{data_type.lower()}.csv"
                self.save_to_csv(processed_data, filename)

def main():
    """Example usage of the CDECDataCollector."""
    # Initialize collector with data directory in the project root
    collector = CDECDataCollector(output_dir="data/cdec")
    
    # Test with Shasta Dam (SHA) which has extensive historical data
    station_id = "SHA"
    
    # Test with a shorter period first (1 month)
    start_date = "2010-01-01"
    end_date = "2010-02-01"
    
    print(f"Collecting data for station {station_id} from {start_date} to {end_date}")
    
    collector.collect_station_history(
        station_id=station_id,
        data_types=['PRECIPITATION', 'STORAGE', 'TEMPERATURE'],  # Changed to more relevant sensors for Shasta Dam
        start_date=start_date,
        end_date=end_date,
        output_prefix='historical_'
    )
    
    print("\nData collection complete. Check the data/cdec directory for output files.")

if __name__ == "__main__":
    main()
