import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

class CMSDataAPI:
    def __init__(self):
        self.catalog_url = "https://data.cms.gov/data.json"
        self.data_dir = Path(__file__).parent / "raw"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_latest_dataset_url(self, title: str) -> Optional[str]:
        """Get the latest dataset URL from the data catalog"""
        try:
            print(f"Fetching data catalog from {self.catalog_url}...")
            response = requests.get(self.catalog_url)
            
            if response.ok:
                catalog = response.json()
                print(f"Found {len(catalog['dataset'])} datasets in catalog")
                
                # Look for our dataset
                for dataset in catalog['dataset']:
                    if dataset['title'] == title:
                        print(f"\nFound matching dataset: {title}")
                        print("Available distributions:")
                        for distro in dataset['distribution']:
                            print(f"- Format: {distro.get('format', 'N/A')}, "
                                  f"Description: {distro.get('description', 'N/A')}, "
                                  f"URL: {distro.get('accessURL', 'N/A')}")
                            
                            if ('format' in distro and 
                                'description' in distro and 
                                distro['format'] == "API" and 
                                distro['description'] == "latest"):
                                print(f"\nUsing API URL: {distro['accessURL']}")
                                return distro['accessURL']
                
                print(f"\nWarning: Could not find dataset with title '{title}'")
                return None
            else:
                print(f"Error fetching catalog: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error getting dataset URL: {str(e)}")
            return None

    def _fetch_filtered_data(self, url: str, filename: str, state: str = 'CA', max_records: int = 10000) -> Optional[pd.DataFrame]:
        """Fetch data with filtering and limits"""
        try:
            print(f"\nFetching data from {url}")
            
            # Add state filter and limit
            filtered_url = f"{url}?filter[nppes_provider_state]={state}&size={max_records}"
            print(f"Using filtered URL: {filtered_url}")
            
            response = requests.get(filtered_url)
            if response.ok:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    print(f"Retrieved {len(data)} records")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    
                    # Save raw data
                    output_path = self.data_dir / filename
                    df.to_csv(output_path, index=False)
                    print(f"Data saved to {output_path}")
                    
                    return df
                else:
                    print(f"Warning: No data returned or unexpected format: {data}")
                    return None
            else:
                print(f"Error fetching data: {response.status_code}")
                print(f"Response: {response.text}")
                return None
            
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None

    def fetch_physician_payment_data(self, state: str = 'CA') -> Optional[pd.DataFrame]:
        """
        Fetch Medicare Physician & Other Practitioners data for a specific state
        """
        title = "Medicare Physician & Other Practitioners - by Provider and Service"
        print(f"\nFetching {title} for state {state}...")
        
        dataset_url = self._get_latest_dataset_url(title)
        if dataset_url:
            return self._fetch_filtered_data(dataset_url, f"physician_payment_data_{state}.csv", state)
        return None

    def fetch_insurance_plans(self, state: str = 'CA') -> Optional[pd.DataFrame]:
        """
        Fetch Medicare Fee-For-Service data for a specific state
        """
        title = "Medicare Fee-For-Service Public Provider Enrollment"
        print(f"\nFetching {title} for state {state}...")
        
        dataset_url = self._get_latest_dataset_url(title)
        if dataset_url:
            return self._fetch_filtered_data(dataset_url, f"insurance_plans_{state}.csv", state)
        return None

    def fetch_hospital_costs(self, state: str = 'CA') -> Optional[pd.DataFrame]:
        """
        Fetch Medicare Inpatient Hospitals data for a specific state
        """
        title = "Medicare Inpatient Hospitals - by Provider and Service"
        print(f"\nFetching {title} for state {state}...")
        
        dataset_url = self._get_latest_dataset_url(title)
        if dataset_url:
            return self._fetch_filtered_data(dataset_url, f"hospital_costs_{state}.csv", state)
        return None

def main():
    """Test the CMS API integration"""
    api = CMSDataAPI()
    state = 'CA'  # Test with California data
    
    # Fetch different types of data
    print(f"\nFetching data for state: {state}")
    
    print("\nFetching physician data...")
    physician_data = api.fetch_physician_payment_data(state)
    
    print("\nFetching insurance data...")
    insurance_data = api.fetch_insurance_plans(state)
    
    print("\nFetching hospital data...")
    hospital_data = api.fetch_hospital_costs(state)
    
    # Print summary of fetched data
    print("\nData Collection Summary:")
    print("----------------------")
    
    if physician_data is not None:
        print(f"Physician Payment Data ({state}): {len(physician_data)} records")
        print("Columns:", ", ".join(physician_data.columns))
        
    if insurance_data is not None:
        print(f"\nInsurance Plan Data ({state}): {len(insurance_data)} records")
        print("Columns:", ", ".join(insurance_data.columns))
        
    if hospital_data is not None:
        print(f"\nHospital Cost Data ({state}): {len(hospital_data)} records")
        print("Columns:", ", ".join(hospital_data.columns))

if __name__ == "__main__":
    main()
