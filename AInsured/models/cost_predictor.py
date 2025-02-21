import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class HealthcareCostPredictor:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.model_dir = Path(__file__).parent
        self.model_path = self.model_dir / "cost_prediction_model.joblib"
        self.scaler_path = self.model_dir / "cost_prediction_scaler.joblib"
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for cost prediction model
        """
        # Geographic features
        features = []
        
        if 'BENE_GEO_CD' in df.columns:
            features.append('BENE_GEO_CD')
            
        # Demographic features
        demographic_cols = [
            'BENE_AVG_AGE',
            'BENE_FEML_PCT',
            'BENE_MALE_PCT',
            'BENE_RACE_WHT_PCT',
            'BENE_RACE_BLACK_PCT',
            'BENE_RACE_HSPNC_PCT',
            'BENE_RACE_OTHR_PCT',
            'BENE_DUAL_PCT',
            'BENE_AVG_RISK_SCRE'
        ]
        features.extend([col for col in demographic_cols if col in df.columns])
        
        # Utilization features
        utilization_cols = [
            'IP_CVRD_STAYS_PER_1000_BENES',
            'ER_VISITS_PER_1000_BENES',
            'OP_VISITS_PER_1000_BENES',
            'HH_VISITS_PER_1000_BENES'
        ]
        features.extend([col for col in utilization_cols if col in df.columns])
        
        # Health condition indicators
        condition_cols = [
            'PQI03_DBTS_AGE_65_74',     # Diabetes
            'PQI08_CHF_AGE_65_74',      # Heart Failure
            'PQI05_COPD_ASTHMA_AGE_65_74',  # COPD
            'PQI07_HYPRTNSN_AGE_65_74'  # Hypertension
        ]
        features.extend([col for col in condition_cols if col in df.columns])
        
        return df[features]

    def prepare_target(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Prepare target variable (total cost per beneficiary)
        """
        if 'TOT_MDCR_PYMT_PC' in df.columns:
            return df['TOT_MDCR_PYMT_PC']
        return None

    def train(self, retrain: bool = False) -> None:
        """
        Train the cost prediction model
        """
        # Check if model already exists
        if not retrain and self.model_path.exists() and self.scaler_path.exists():
            print("Loading existing model...")
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return
            
        print("Training new cost prediction model...")
        
        # Load geographic variation data
        data_path = self.data_dir / "processed" / "geographic_variation_processed.csv"
        if not data_path.exists():
            raise FileNotFoundError("Geographic variation data not found")
            
        df = pd.read_csv(data_path)
        
        # Prepare features and target
        X = self.prepare_features(df)
        y = self.prepare_target(df)
        
        if X is None or y is None:
            raise ValueError("Could not prepare features or target")
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Mean Squared Error: ${mse:,.2f}")
        print(f"R-squared Score: {r2:.4f}")
        
        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
    def predict_cost(self, 
                    age: int,
                    gender: str,
                    state: str,
                    has_chronic_conditions: bool,
                    utilization_level: str = 'medium',
                    dual_eligible: bool = False) -> Dict[str, any]:
        """
        Predict healthcare costs for an individual
        """
        if self.model is None:
            self.train()
            
        # Create feature vector
        features = pd.DataFrame({
            'BENE_AVG_AGE': [age],
            'BENE_FEML_PCT': [1.0 if gender.lower() == 'female' else 0.0],
            'BENE_MALE_PCT': [1.0 if gender.lower() == 'male' else 0.0],
            'BENE_DUAL_PCT': [1.0 if dual_eligible else 0.0],
            'BENE_AVG_RISK_SCRE': [1.5 if has_chronic_conditions else 1.0]
        })
        
        # Add utilization features
        utilization_levels = {
            'low': {'ip': 200, 'er': 300, 'op': 2000, 'hh': 1000},
            'medium': {'ip': 400, 'er': 600, 'op': 4000, 'hh': 2000},
            'high': {'ip': 800, 'er': 1200, 'op': 8000, 'hh': 4000}
        }
        level = utilization_levels.get(utilization_level.lower(), utilization_levels['medium'])
        
        features['IP_CVRD_STAYS_PER_1000_BENES'] = level['ip']
        features['ER_VISITS_PER_1000_BENES'] = level['er']
        features['OP_VISITS_PER_1000_BENES'] = level['op']
        features['HH_VISITS_PER_1000_BENES'] = level['hh']
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        predicted_cost = self.model.predict(features_scaled)[0]
        
        # Get feature importances
        feature_importance = dict(zip(features.columns, self.model.feature_importances_))
        
        return {
            'predicted_annual_cost': predicted_cost,
            'confidence_interval': (
                predicted_cost * 0.8,  # Lower bound
                predicted_cost * 1.2   # Upper bound
            ),
            'feature_importance': feature_importance,
            'risk_factors': {
                'age': 'high' if age >= 75 else 'medium' if age >= 65 else 'low',
                'chronic_conditions': 'high' if has_chronic_conditions else 'low',
                'utilization': utilization_level,
                'dual_eligible': 'yes' if dual_eligible else 'no'
            }
        }

def main():
    """Train and test the healthcare cost prediction model"""
    predictor = HealthcareCostPredictor()
    
    # Train model
    print("Training model...")
    predictor.train(retrain=True)
    
    # Test predictions
    test_cases = [
        {
            'age': 70,
            'gender': 'female',
            'state': 'CA',
            'has_chronic_conditions': True,
            'utilization_level': 'medium',
            'dual_eligible': False
        },
        {
            'age': 65,
            'gender': 'male',
            'state': 'FL',
            'has_chronic_conditions': False,
            'utilization_level': 'low',
            'dual_eligible': False
        },
        {
            'age': 80,
            'gender': 'female',
            'state': 'NY',
            'has_chronic_conditions': True,
            'utilization_level': 'high',
            'dual_eligible': True
        }
    ]
    
    print("\nTesting predictions:")
    for case in test_cases:
        prediction = predictor.predict_cost(**case)
        print(f"\nPrediction for {case['age']} year old {case['gender']} in {case['state']}:")
        print(f"Predicted Annual Cost: ${prediction['predicted_annual_cost']:,.2f}")
        print(f"Confidence Interval: ${prediction['confidence_interval'][0]:,.2f} - ${prediction['confidence_interval'][1]:,.2f}")
        print("Risk Factors:", prediction['risk_factors'])
        print("\nTop Feature Importance:")
        sorted_features = sorted(
            prediction['feature_importance'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for feature, importance in sorted_features[:5]:
            print(f"{feature}: {importance:.4f}")

if __name__ == "__main__":
    main()
