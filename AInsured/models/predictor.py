import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

class HealthCostPredictor:
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / "data"
        self.payment_data = None
        self.plan_data = None
        self.load_data()
        
    def load_data(self):
        """Load processed CMS data"""
        payment_path = self.data_dir / "processed" / "processed_medicare_payment_data.csv"
        plan_path = self.data_dir / "raw" / "medicare_plan_data.csv"
        
        try:
            if payment_path.exists():
                self.payment_data = pd.read_csv(payment_path)
            if plan_path.exists():
                self.plan_data = pd.read_csv(plan_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def predict_annual_cost(self, health_info: Dict) -> float:
        """
        Predict annual healthcare costs based on user health information
        """
        # TODO: Implement actual prediction logic
        # This is a placeholder implementation
        base_cost = 5000.0
        
        # Age factor
        age_factor = health_info['age'] / 50.0
        
        # BMI factor (assuming normal BMI is 18.5-24.9)
        bmi = health_info['bmi']
        if bmi < 18.5 or bmi > 24.9:
            bmi_factor = 1.2
        else:
            bmi_factor = 1.0
            
        # Smoker factor
        smoker_factor = 1.5 if health_info['smoker'] else 1.0
        
        # Exercise factor
        exercise_freq = health_info.get('exercise_frequency', 0)
        exercise_factor = 1.0 - (min(exercise_freq, 7) * 0.02)  # Up to 14% reduction
        
        # Chronic conditions factor
        conditions = health_info.get('chronic_conditions', [])
        condition_factor = 1.0 + (len(conditions) * 0.1)  # 10% increase per condition
        
        # Calculate final cost
        predicted_cost = (
            base_cost * 
            age_factor * 
            bmi_factor * 
            smoker_factor * 
            exercise_factor * 
            condition_factor
        )
        
        return round(predicted_cost, 2)
    
    def recommend_plans(self, predicted_cost: float, health_info: Dict) -> List[Dict]:
        """
        Recommend insurance plans based on predicted costs and health info
        """
        # TODO: Implement actual plan recommendation logic
        # This is a placeholder implementation
        plans = [
            {
                "plan_id": "basic_plan",
                "name": "Basic Coverage",
                "premium": 300.0,
                "deductible": 2000.0,
                "out_of_pocket_max": 8000.0,
                "coverage_level": "Silver"
            },
            {
                "plan_id": "premium_plan",
                "name": "Premium Coverage",
                "premium": 500.0,
                "deductible": 1000.0,
                "out_of_pocket_max": 5000.0,
                "coverage_level": "Gold"
            }
        ]
        
        return plans
    
    def calculate_confidence_score(self, health_info: Dict) -> float:
        """
        Calculate confidence score for the prediction
        """
        # TODO: Implement actual confidence calculation
        # This is a placeholder implementation
        base_confidence = 0.85
        
        # Reduce confidence for extreme ages
        age = health_info['age']
        if age < 25 or age > 75:
            base_confidence *= 0.9
            
        # Reduce confidence for multiple chronic conditions
        conditions = health_info.get('chronic_conditions', [])
        if len(conditions) > 2:
            base_confidence *= 0.95
            
        return round(base_confidence, 2)
    
    def get_prediction(self, health_info: Dict) -> Dict:
        """
        Get complete prediction including costs, recommended plans, and confidence score
        """
        predicted_cost = self.predict_annual_cost(health_info)
        recommended_plans = self.recommend_plans(predicted_cost, health_info)
        confidence_score = self.calculate_confidence_score(health_info)
        
        return {
            "expected_annual_cost": predicted_cost,
            "recommended_plans": recommended_plans,
            "confidence_score": confidence_score
        }
