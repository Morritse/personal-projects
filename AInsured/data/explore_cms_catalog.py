import requests
import json
from typing import List, Dict
from pathlib import Path
import time
from collections import defaultdict

class CMSCatalogExplorer:
    def __init__(self):
        self.catalog_url = "https://data.cms.gov/data.json"
        self.data_dir = Path(__file__).parent
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Track progress
        self.progress_file = self.raw_dir / "catalog_progress.json"
        
        # Categories of interest with related keywords
        self.categories = {
            'claims_data': [
                'claims', 'billing', 'payment', 'cost'
            ],
            'insurance_plans': [
                'insurance', 'plan', 'medicare advantage', 'part d'
            ],
            'healthcare_costs': [
                'expenditure', 'spending', 'charge', 'fee'
            ]
        }

    def process_chunk(self, datasets: List[Dict], start_idx: int) -> Dict:
        """
        Process a chunk of datasets
        """
        chunk_results = defaultdict(list)
        
        print(f"\nProcessing datasets {start_idx} to {start_idx + len(datasets)}")
        for dataset in datasets:
            title = dataset.get('title', '')
            desc = dataset.get('description', '')
            content = f"{title} {desc}".lower()
            
            # Check for API access
            has_api = False
            api_url = None
            for dist in dataset.get('distribution', []):
                if dist.get('format') == 'API':
                    has_api = True
                    api_url = dist.get('accessURL')
                    break
            
            if not has_api:
                continue
                
            # Categorize dataset
            for category, keywords in self.categories.items():
                matches = [kw for kw in keywords if kw.lower() in content]
                if matches:
                    chunk_results[category].append({
                        'title': title,
                        'description': desc[:200],
                        'api_url': api_url,
                        'matching_keywords': matches
                    })
                    print(f"\nFound relevant dataset: {title}")
                    print(f"Category: {category}")
                    print(f"Matching keywords: {', '.join(matches)}")
        
        return dict(chunk_results)

    def analyze_catalog(self, chunk_size: int = 500):
        """
        Analyze the CMS data catalog in chunks
        """
        try:
            # Check for existing progress
            start_idx = 0
            results = defaultdict(list)
            
            if self.progress_file.exists():
                with open(self.progress_file) as f:
                    progress = json.load(f)
                    start_idx = progress['next_index']
                    results = defaultdict(list, progress['results'])
                print(f"Resuming from dataset {start_idx}")
            
            # Fetch catalog if needed
            if start_idx == 0:
                print("Fetching CMS data catalog...")
                response = requests.get(self.catalog_url)
                
                if response.ok:
                    catalog = response.json()
                    datasets = catalog.get('dataset', [])
                    
                    print(f"\nFound {len(datasets)} total datasets")
                    print(f"Processing in chunks of {chunk_size}")
                    
                    # Process chunks
                    while start_idx < len(datasets):
                        chunk = datasets[start_idx:start_idx + chunk_size]
                        chunk_results = self.process_chunk(chunk, start_idx)
                        
                        # Merge chunk results
                        for category, datasets in chunk_results.items():
                            results[category].extend(datasets)
                        
                        # Save progress
                        progress = {
                            'next_index': start_idx + chunk_size,
                            'results': dict(results)
                        }
                        with open(self.progress_file, 'w') as f:
                            json.dump(progress, f, indent=2)
                        
                        start_idx += chunk_size
                        
                        # Summary after each chunk
                        print("\nCurrent Progress:")
                        for category, datasets in results.items():
                            print(f"{category}: {len(datasets)} datasets")
                        
                        proceed = input("\nProcess next chunk? (y/n): ")
                        if proceed.lower() != 'y':
                            break
                    
                    # Save final results
                    final_results = {
                        'summary': {
                            'total_processed': start_idx,
                            'datasets_by_category': {
                                cat: len(ds) for cat, ds in results.items()
                            }
                        },
                        'datasets': dict(results)
                    }
                    
                    results_path = self.raw_dir / "cms_catalog_analysis.json"
                    with open(results_path, 'w') as f:
                        json.dump(final_results, f, indent=2)
                    print(f"\nSaved analysis to {results_path}")
                    
                    return final_results
                else:
                    print(f"Error fetching catalog: {response.status_code}")
                    
        except Exception as e:
            print(f"Error analyzing catalog: {str(e)}")
            
        return None

def main():
    """Analyze CMS Data Catalog in manageable chunks"""
    explorer = CMSCatalogExplorer()
    
    print("Starting Chunked CMS Data Analysis")
    print("================================")
    print("Categories being analyzed:")
    for category, keywords in explorer.categories.items():
        print(f"\n{category}:")
        print(f"Keywords: {', '.join(keywords)}")
    
    results = explorer.analyze_catalog(chunk_size=500)
    
    if results:
        print("\nAnalysis complete!")
        print("\nDatasets found by category:")
        for category, count in results['summary']['datasets_by_category'].items():
            print(f"{category}: {count} datasets")

if __name__ == "__main__":
    main()
