import requests
import json
from pathlib import Path
import time
from typing import Dict, List
import pandas as pd

class CMSDataFetcher:
    def __init__(self):
        self.data_dir = Path(__file__).parent
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Key datasets with verified API endpoints
        self.target_datasets = {
            "geographic_variation": {
                "id": "6219697b-8f6c-4164-bed4-cd9317c58ebc",
                "name": "Medicare Geographic Variation - by National, State & County",
                "description": "Geographic differences in healthcare costs and utilization",
                "key_fields": [
                    "payment_amt", 
                    "utilization",
                    "beneficiary_count"
                ]
            },
            "claims_patterns": {
                "id": "6395b458-2f89-4828-8c1a-e1e16b723d48",
                "name": "Medicare Fee-for-Service Comprehensive Error Rate Testing",
                "description": "Claims and payment pattern data",
                "key_fields": [
                    "claim_type",
                    "payment_amt",
                    "error_rate"
                ]
            },
            "plan_enrollment": {
                "id": "d7fabe1e-d19b-4333-9eff-e80e0643f2fd",
                "name": "Medicare Monthly Enrollment",
                "description": "Medicare Advantage plan enrollment data",
                "key_fields": [
                    "plan_type",
                    "enrollment_count",
                    "contract_number"
                ]
            }
        }

    def fetch_dataset_sample(self, dataset_id: str, limit: int = 50) -> List[Dict]:
        """
        Fetch a sample of records from a dataset
        """
        try:
            base_url = f"https://data.cms.gov/data-api/v1/dataset/{dataset_id}/data"
            params = {
                'size': min(limit, 50),  # Process in smaller chunks
                'offset': 0
            }
            
            print(f"\nFetching {limit} records from dataset {dataset_id}...")
            response = requests.get(base_url, params=params)
            
            if response.ok:
                data = response.json()
                
                # Save raw sample
                sample_path = self.raw_dir / f"{dataset_id}_sample.json"
                with open(sample_path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved sample to {sample_path}")
                
                return data
            else:
                print(f"Error fetching data: {response.status_code}")
                if response.text:
                    print("Response:", response.text[:500])
                
        except Exception as e:
            print(f"Error: {str(e)}")
            
        return None

    def process_dataset(self, dataset_id: str, data: List[Dict]) -> pd.DataFrame:
        """
        Process a dataset into a pandas DataFrame
        """
        if not data:
            return None
            
        df = pd.DataFrame(data)
        
        # Save processed data
        processed_path = self.processed_dir / f"{dataset_id}_processed.csv"
        df.to_csv(processed_path, index=False)
        print(f"Saved processed data to {processed_path}")
        
        return df

    def fetch_all_samples(self):
        """
        Fetch samples from all target datasets
        """
        results = {}
        
        for dataset_id, info in self.target_datasets.items():
            print(f"\nProcessing {info['name']}...")
            print(f"Description: {info['description']}")
            print(f"Key fields: {', '.join(info['key_fields'])}")
            
            # Fetch sample
            data = self.fetch_dataset_sample(info['id'])
            
            if data:
                # Process data
                df = self.process_dataset(dataset_id, data)
                if isinstance(df, pd.DataFrame):
                    results[dataset_id] = {
                        'name': info['name'],
                        'record_count': len(df),
                        'columns': list(df.columns)
                    }
                    
                    # Print sample of data structure
                    print("\nSample data structure:")
                    print(df.head(1).to_string())
            
            # Rate limiting
            time.sleep(2)
            
        return results

def main():
    """Fetch samples of key CMS datasets for insurance cost prediction"""
    fetcher = CMSDataFetcher()
    
    print("Starting CMS Data Collection")
    print("==========================")
    print("Targeting 3 key datasets for insurance cost prediction:")
    print("1. Geographic variation in healthcare costs")
    print("2. Claims and payment patterns")
    print("3. Medicare Advantage plan enrollment")
    
    results = fetcher.fetch_all_samples()
    
    if results:
        print("\nData collection complete!")
        print("\nDataset summaries:")
        for dataset_id, info in results.items():
            print(f"\n{info['name']}:")
            print(f"Records collected: {info['record_count']}")
            print(f"Available fields: {', '.join(info['columns'])}")

if __name__ == "__main__":
    main()
