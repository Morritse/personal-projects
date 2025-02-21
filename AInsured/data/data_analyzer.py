import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class InsuranceCostAnalyzer:
    def __init__(self):
        self.data_dir = Path(__file__).parent
        self.processed_dir = self.data_dir / "processed"
        
    def clean_numeric_column(self, series: pd.Series) -> pd.Series:
        """Clean numeric columns by removing non-numeric characters and converting to float"""
        def clean_value(x):
            if pd.isna(x):
                return np.nan
            try:
                # Remove any non-numeric characters except decimal points and negative signs
                cleaned = ''.join(c for c in str(x) if c.isdigit() or c in '.-')
                return float(cleaned) if cleaned else np.nan
            except:
                return np.nan
        
        return series.apply(clean_value)

    def load_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load and clean all processed datasets
        """
        datasets = {}
        
        # Geographic cost variations
        geo_path = self.processed_dir / "geographic_variation_processed.csv"
        if geo_path.exists():
            df = pd.read_csv(geo_path)
            # Clean numeric columns
            numeric_columns = [
                'TOT_MDCR_PYMT_PC',
                'IP_MDCR_PYMT_PC',
                'OP_MDCR_PYMT_PC',
                'ER_VISITS_PER_1000_BENES'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = self.clean_numeric_column(df[col])
            datasets['geographic'] = df
            
        # Claims patterns
        claims_path = self.processed_dir / "claims_patterns_processed.csv"
        if claims_path.exists():
            datasets['claims'] = pd.read_csv(claims_path)
            
        # Plan enrollment
        enrollment_path = self.processed_dir / "plan_enrollment_processed.csv"
        if enrollment_path.exists():
            df = pd.read_csv(enrollment_path)
            # Clean numeric columns
            numeric_columns = [
                'TOT_BENES',
                'MA_AND_OTH_BENES',
                'PRSCRPTN_DRUG_TOT_BENES',
                'DUAL_TOT_BENES'
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = self.clean_numeric_column(df[col])
            datasets['enrollment'] = df
            
        return datasets
    
    def analyze_regional_costs(self, df: pd.DataFrame) -> Dict:
        """
        Analyze healthcare costs by region
        """
        cost_metrics = {
            'total_payment': 'TOT_MDCR_PYMT_PC',
            'inpatient': 'IP_MDCR_PYMT_PC',
            'outpatient': 'OP_MDCR_PYMT_PC',
            'emergency': 'ER_VISITS_PER_1000_BENES'
        }
        
        results = {}
        for metric, col in cost_metrics.items():
            if col in df.columns:
                # Convert to numeric and handle any remaining invalid values
                values = pd.to_numeric(df[col], errors='coerce')
                valid_values = values.dropna()
                
                if not valid_values.empty:
                    results[metric] = {
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'min': valid_values.min(),
                        'max': valid_values.max()
                    }
                else:
                    print(f"Warning: No valid numeric values found for {metric}")
        
        return results
    
    def analyze_claim_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze common claim types and error rates
        """
        if 'Provider Type' in df.columns:
            claim_patterns = {
                'provider_types': df['Provider Type'].value_counts().to_dict(),
                'error_rates': df['Error Code'].value_counts().to_dict() if 'Error Code' in df.columns else {}
            }
            return claim_patterns
        return {}
    
    def analyze_plan_enrollment(self, df: pd.DataFrame) -> Dict:
        """
        Analyze insurance plan enrollment patterns
        """
        enrollment_metrics = {
            'total_beneficiaries': 'TOT_BENES',
            'ma_enrollment': 'MA_AND_OTH_BENES',
            'prescription_drug': 'PRSCRPTN_DRUG_TOT_BENES',
            'dual_eligible': 'DUAL_TOT_BENES'
        }
        
        results = {}
        for metric, col in enrollment_metrics.items():
            if col in df.columns:
                # Convert to numeric and handle any remaining invalid values
                values = pd.to_numeric(df[col], errors='coerce')
                valid_values = values.dropna()
                
                if not valid_values.empty:
                    results[metric] = {
                        'mean': valid_values.mean(),
                        'std': valid_values.std(),
                        'total': valid_values.sum()
                    }
                else:
                    print(f"Warning: No valid numeric values found for {metric}")
        
        return results

    def predict_costs(self, 
                     age: int,
                     state: str,
                     has_chronic_conditions: bool,
                     income_level: str) -> Dict:
        """
        Predict expected healthcare costs based on user factors
        """
        datasets = self.load_datasets()
        
        # Base prediction on geographic data
        if 'geographic' in datasets:
            geo_df = datasets['geographic']
            
            # Get regional cost factors
            regional_costs = self.analyze_regional_costs(geo_df)
            base_cost = regional_costs.get('total_payment', {}).get('mean', 0)
            
            if base_cost == 0:
                print("Warning: Could not determine base cost from data")
                base_cost = 10000  # Fallback to reasonable default
            
            # Age adjustment
            age_factors = {
                range(0, 25): 0.5,    # Lower costs for young
                range(25, 45): 0.7,
                range(45, 65): 1.0,
                range(65, 75): 1.3,
                range(75, 85): 1.5,
                range(85, 200): 1.8   # Higher costs for elderly
            }
            
            age_factor = 1.0
            for age_range, factor in age_factors.items():
                if age in age_range:
                    age_factor = factor
                    break
            
            # Chronic condition adjustment
            condition_factor = 1.5 if has_chronic_conditions else 1.0
            
            # Income level adjustment (affects out-of-pocket)
            income_factors = {
                'low': 0.8,    # More likely to qualify for assistance
                'medium': 1.0,
                'high': 1.2    # Higher out-of-pocket
            }
            income_factor = income_factors.get(income_level, 1.0)
            
            # Calculate predicted costs
            predicted_annual_cost = base_cost * age_factor * condition_factor * income_factor
            
            return {
                'predicted_annual_cost': predicted_annual_cost,
                'confidence_interval': (predicted_annual_cost * 0.8, predicted_annual_cost * 1.2),
                'factors': {
                    'age_impact': age_factor,
                    'health_impact': condition_factor,
                    'income_impact': income_factor,
                    'regional_variation': regional_costs
                }
            }
            
        return {}

    def recommend_plans(self,
                       age: int,
                       state: str,
                       predicted_costs: float,
                       has_chronic_conditions: bool) -> List[Dict]:
        """
        Recommend insurance plans based on predicted costs
        """
        datasets = self.load_datasets()
        
        if 'enrollment' in datasets:
            enrollment_df = datasets['enrollment']
            enrollment_patterns = self.analyze_plan_enrollment(enrollment_df)
            
            # Basic plan types
            plan_types = [
                {
                    'type': 'Original Medicare',
                    'description': 'Traditional fee-for-service Medicare (Parts A & B)',
                    'best_for': ['flexibility in choosing providers', 'no network restrictions'],
                    'considerations': ['higher out-of-pocket costs', 'no drug coverage']
                },
                {
                    'type': 'Medicare Advantage',
                    'description': 'All-in-one alternative to Original Medicare',
                    'best_for': ['predictable costs', 'additional benefits'],
                    'considerations': ['network restrictions', 'regional availability']
                },
                {
                    'type': 'Medigap + Part D',
                    'description': 'Original Medicare + supplemental coverage',
                    'best_for': ['comprehensive coverage', 'chronic conditions'],
                    'considerations': ['higher premiums', 'separate drug plan needed']
                }
            ]
            
            # Score plans based on user factors
            scored_plans = []
            for plan in plan_types:
                score = 0
                
                if age >= 65:
                    score += 1  # All plans suitable for 65+
                
                if has_chronic_conditions:
                    if plan['type'] == 'Medigap + Part D':
                        score += 2  # Better for chronic conditions
                    elif plan['type'] == 'Medicare Advantage':
                        score += 1  # Care coordination helpful
                
                if predicted_costs > 10000:
                    if plan['type'] == 'Medigap + Part D':
                        score += 2  # Better for high costs
                
                plan['score'] = score
                scored_plans.append(plan)
            
            # Sort by score and return top recommendations
            scored_plans.sort(key=lambda x: x['score'], reverse=True)
            return scored_plans
            
        return []

def main():
    """Analyze healthcare costs and recommend insurance plans"""
    analyzer = InsuranceCostAnalyzer()
    
    print("Loading and analyzing healthcare data...")
    datasets = analyzer.load_datasets()
    
    if datasets:
        print("\nAnalyzing regional cost variations:")
        if 'geographic' in datasets:
            regional_costs = analyzer.analyze_regional_costs(datasets['geographic'])
            for metric, stats in regional_costs.items():
                print(f"\n{metric}:")
                for stat, value in stats.items():
                    print(f"  {stat}: ${value:,.2f}")
        
        print("\nAnalyzing enrollment patterns:")
        if 'enrollment' in datasets:
            enrollment_patterns = analyzer.analyze_plan_enrollment(datasets['enrollment'])
            for metric, stats in enrollment_patterns.items():
                print(f"\n{metric}:")
                for stat, value in stats.items():
                    print(f"  {stat}: {value:,.2f}")
        
        # Example prediction
        print("\nGenerating example cost prediction...")
        example_prediction = analyzer.predict_costs(
            age=70,
            state="CA",
            has_chronic_conditions=True,
            income_level="medium"
        )
        
        if example_prediction:
            print("\nExample Cost Prediction:")
            print(f"Predicted Annual Cost: ${example_prediction['predicted_annual_cost']:,.2f}")
            print(f"Confidence Interval: ${example_prediction['confidence_interval'][0]:,.2f} - ${example_prediction['confidence_interval'][1]:,.2f}")
            
            # Get plan recommendations
            print("\nGenerating plan recommendations...")
            recommendations = analyzer.recommend_plans(
                age=70,
                state="CA",
                predicted_costs=example_prediction['predicted_annual_cost'],
                has_chronic_conditions=True
            )
            
            if recommendations:
                print("\nRecommended Insurance Plans:")
                for i, plan in enumerate(recommendations, 1):
                    print(f"\n{i}. {plan['type']} (Score: {plan['score']})")
                    print(f"Description: {plan['description']}")
                    print(f"Best for: {', '.join(plan['best_for'])}")
                    print(f"Considerations: {', '.join(plan['considerations'])}")

if __name__ == "__main__":
    main()
