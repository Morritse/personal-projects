from data.cms_api import CMSDataAPI
from data.data_processor import CMSDataProcessor

def main():
    print("Testing CMS Data Integration")
    print("===========================")
    
    # Step 1: Fetch Data
    print("\nStep 1: Fetching CMS Data")
    print("--------------------------")
    api = CMSDataAPI()
    
    # Fetch different types of data
    physician_data = api.fetch_physician_payment_data()
    insurance_data = api.fetch_insurance_plans()
    hospital_data = api.fetch_hospital_costs()
    
    # Step 2: Process Data
    print("\nStep 2: Processing Data")
    print("----------------------")
    processor = CMSDataProcessor()
    
    # Process each dataset
    processed_physician = processor.process_physician_data()
    processed_insurance = processor.process_insurance_plans()
    processed_hospital = processor.process_hospital_costs()
    
    # Step 3: Combine Data
    print("\nStep 3: Combining Datasets")
    print("-------------------------")
    if all([processed_physician is not None,
            processed_insurance is not None,
            processed_hospital is not None]):
        combined_data = processor.combine_datasets()
        
        if combined_data is not None:
            print("\nFinal Dataset Summary:")
            print("---------------------")
            print(f"Total Records: {len(combined_data)}")
            print(f"Features: {len(combined_data.columns)}")
            print("\nSample of combined data:")
            print(combined_data.head())
    else:
        print("Error: Could not create combined dataset")

if __name__ == "__main__":
    main()
