import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class CMSDataProcessor:
    def __init__(self):
        self.data_dir = Path(__file__).parent
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def process_physician_data(self, year: int = 2022) -> Optional[pd.DataFrame]:
        """Process physician payment data"""
        input_path = self.raw_dir / f"physician_payment_data_{year}.csv"
        
        if not input_path.exists():
            print(f"No physician payment data found for {year}")
            return None
            
        try:
            df = pd.read_csv(input_path)
            
            # Clean and process data
            processed = df.copy()
            
            # Convert numeric columns
            numeric_columns = [
                'average_Medicare_allowed_amt',
                'average_submitted_chrg_amt',
                'average_Medicare_payment_amt'
            ]
            
            for col in numeric_columns:
                if col in processed.columns:
                    processed[col] = pd.to_numeric(processed[col], errors='coerce')
            
            # Group by provider and specialty
            grouped = processed.groupby(['provider_type', 'nppes_provider_state']).agg({
                'average_Medicare_allowed_amt': 'mean',
                'average_submitted_chrg_amt': 'mean',
                'average_Medicare_payment_amt': 'mean',
                'npi': 'count'  # Count of procedures
            }).reset_index()
            
            # Save processed data
            output_path = self.processed_dir / f"processed_physician_data_{year}.csv"
            grouped.to_csv(output_path, index=False)
            print(f"Processed physician data saved to {output_path}")
            
            return grouped
            
        except Exception as e:
            print(f"Error processing physician data: {str(e)}")
            return None
    
    def process_insurance_plans(self) -> Optional[pd.DataFrame]:
        """Process insurance plan data"""
        input_path = self.raw_dir / "insurance_plans.csv"
        
        if not input_path.exists():
            print("No insurance plan data found")
            return None
            
        try:
            df = pd.read_csv(input_path)
            
            # Clean and process data
            processed = df.copy()
            
            # Save processed data
            output_path = self.processed_dir / "processed_insurance_plans.csv"
            processed.to_csv(output_path, index=False)
            print(f"Processed insurance plan data saved to {output_path}")
            
            return processed
            
        except Exception as e:
            print(f"Error processing insurance plan data: {str(e)}")
            return None
    
    def process_hospital_costs(self) -> Optional[pd.DataFrame]:
        """Process hospital cost data"""
        input_path = self.raw_dir / "hospital_costs.csv"
        
        if not input_path.exists():
            print("No hospital cost data found")
            return None
            
        try:
            df = pd.read_csv(input_path)
            
            # Clean and process data
            processed = df.copy()
            
            # Save processed data
            output_path = self.processed_dir / "processed_hospital_costs.csv"
            processed.to_csv(output_path, index=False)
            print(f"Processed hospital cost data saved to {output_path}")
            
            return processed
            
        except Exception as e:
            print(f"Error processing hospital cost data: {str(e)}")
            return None
    
    def combine_datasets(self) -> Optional[pd.DataFrame]:
        """Combine all processed datasets for analysis"""
        try:
            # Load processed datasets
            physician_data = pd.read_csv(self.processed_dir / "processed_physician_data_2022.csv")
            insurance_data = pd.read_csv(self.processed_dir / "processed_insurance_plans.csv")
            hospital_data = pd.read_csv(self.processed_dir / "processed_hospital_costs.csv")
            
            # Combine datasets based on state
            # This is a simplified example - actual implementation would need
            # more sophisticated joining logic based on available columns
            combined = pd.merge(
                physician_data,
                insurance_data,
                left_on='nppes_provider_state',
                right_on='state',
                how='left'
            )
            
            combined = pd.merge(
                combined,
                hospital_data,
                left_on='nppes_provider_state',
                right_on='state',
                how='left'
            )
            
            # Save combined dataset
            output_path = self.processed_dir / "combined_healthcare_data.csv"
            combined.to_csv(output_path, index=False)
            print(f"Combined dataset saved to {output_path}")
            
            return combined
            
        except Exception as e:
            print(f"Error combining datasets: {str(e)}")
            return None

def main():
    """Test the data processing pipeline"""
    processor = CMSDataProcessor()
    
    # Process each dataset
    physician_data = processor.process_physician_data()
    insurance_data = processor.process_insurance_plans()
    hospital_data = processor.process_hospital_costs()
    
    # Combine datasets
    if all([physician_data is not None, 
            insurance_data is not None, 
            hospital_data is not None]):
        combined_data = processor.combine_datasets()
        
        if combined_data is not None:
            print("\nData Processing Summary:")
            print("----------------------")
            print(f"Combined Dataset Shape: {combined_data.shape}")
            print("\nColumns:", ", ".join(combined_data.columns))
    else:
        print("\nError: Could not combine datasets due to missing data")

if __name__ == "__main__":
    main()
