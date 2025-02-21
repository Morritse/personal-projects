from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class HealthProfile:
    age: int
    bmi: float
    smoker: bool
    gender: str
    exercise_frequency: int  # days per week
    chronic_conditions: List[str] = None
    family_history: List[str] = None

class HealthRiskAssessor:
    def __init__(self):
        # Common chronic conditions and their relative risk factors
        self.chronic_conditions = {
            "diabetes": {
                "related_procedures": ["A1C test", "foot exam", "eye exam"],
                "risk_factor": 1.5,
                "common_drgs": ["638", "637", "636"]  # Diabetes DRG codes
            },
            "hypertension": {
                "related_procedures": ["blood pressure screening", "cardiac exam"],
                "risk_factor": 1.3,
                "common_drgs": ["302", "303", "304"]  # Hypertension DRG codes
            },
            "asthma": {
                "related_procedures": ["pulmonary function test", "peak flow measurement"],
                "risk_factor": 1.2,
                "common_drgs": ["202", "203"]  # Respiratory DRG codes
            }
        }
        
        # Age-based risk factors (per decade)
        self.age_risk_factors = {
            0: 1.0,   # 0-9
            10: 1.1,  # 10-19
            20: 1.2,  # 20-29
            30: 1.3,  # 30-39
            40: 1.5,  # 40-49
            50: 1.8,  # 50-59
            60: 2.2,  # 60-69
            70: 2.7,  # 70-79
            80: 3.2   # 80+
        }
        
        # BMI risk factors
        self.bmi_categories = {
            "underweight": {"range": (0, 18.5), "risk_factor": 1.2},
            "normal": {"range": (18.5, 25), "risk_factor": 1.0},
            "overweight": {"range": (25, 30), "risk_factor": 1.3},
            "obese": {"range": (30, float('inf')), "risk_factor": 1.6}
        }
        
        # Common preventive services by age/gender
        self.preventive_services = {
            "all": [
                "annual physical",
                "blood pressure screening",
                "cholesterol screening",
                "diabetes screening"
            ],
            "female": {
                "21+": ["cervical cancer screening"],
                "40+": ["mammogram"],
                "50+": ["colorectal cancer screening"]
            },
            "male": {
                "50+": ["colorectal cancer screening", "prostate cancer screening"]
            }
        }
    
    def _get_age_risk(self, age: int) -> float:
        """Calculate age-based risk factor"""
        decade = (age // 10) * 10
        return self.age_risk_factors.get(decade, 3.2)  # Use 80+ factor for ages 90+
        
    def _get_bmi_risk(self, bmi: float) -> float:
        """Calculate BMI-based risk factor"""
        for category in self.bmi_categories.values():
            if category["range"][0] <= bmi < category["range"][1]:
                return category["risk_factor"]
        return 1.0
        
    def _get_lifestyle_risk(self, profile: HealthProfile) -> float:
        """Calculate lifestyle-based risk factor"""
        risk = 1.0
        
        # Smoking risk
        if profile.smoker:
            risk *= 1.5
            
        # Exercise benefit (up to 20% reduction in risk)
        exercise_benefit = min(0.2, profile.exercise_frequency * 0.03)
        risk *= (1 - exercise_benefit)
        
        return risk
        
    def get_recommended_services(self, profile: HealthProfile) -> List[str]:
        """Get recommended preventive services based on health profile"""
        services = self.preventive_services["all"].copy()
        
        # Add gender-specific services
        gender_services = self.preventive_services.get(profile.gender.lower(), {})
        for age_group, age_services in gender_services.items():
            min_age = int(age_group.replace("+", ""))
            if profile.age >= min_age:
                services.extend(age_services)
                
        # Add condition-specific services
        if profile.chronic_conditions:
            for condition in profile.chronic_conditions:
                if condition in self.chronic_conditions:
                    services.extend(self.chronic_conditions[condition]["related_procedures"])
                    
        return list(set(services))  # Remove duplicates
        
    def assess_health_risk(self, profile: HealthProfile) -> Dict:
        """
        Assess overall health risk and recommend services
        Returns risk factors and recommended services
        """
        # Calculate component risks
        age_risk = self._get_age_risk(profile.age)
        bmi_risk = self._get_bmi_risk(profile.bmi)
        lifestyle_risk = self._get_lifestyle_risk(profile)
        
        # Calculate condition risks
        condition_risk = 1.0
        if profile.chronic_conditions:
            for condition in profile.chronic_conditions:
                if condition in self.chronic_conditions:
                    condition_risk *= self.chronic_conditions[condition]["risk_factor"]
        
        # Calculate overall risk score (1.0 = average risk)
        overall_risk = age_risk * bmi_risk * lifestyle_risk * condition_risk
        
        # Get recommended services
        services = self.get_recommended_services(profile)
        
        # Prepare detailed assessment
        assessment = {
            "risk_score": round(overall_risk, 2),
            "risk_factors": {
                "age": round(age_risk, 2),
                "bmi": round(bmi_risk, 2),
                "lifestyle": round(lifestyle_risk, 2),
                "conditions": round(condition_risk, 2)
            },
            "risk_level": "low" if overall_risk < 1.3 else "medium" if overall_risk < 2.0 else "high",
            "recommended_services": services,
            "health_insights": []
        }
        
        # Add specific insights based on risk factors
        if profile.smoker:
            assessment["health_insights"].append(
                "Smoking significantly increases health risks. Consider smoking cessation programs."
            )
        
        if profile.bmi >= 30:
            assessment["health_insights"].append(
                "BMI indicates obesity. Consider weight management and regular exercise."
            )
            
        if profile.exercise_frequency < 3:
            assessment["health_insights"].append(
                "Increased physical activity (aim for 3+ days/week) could reduce health risks."
            )
            
        return assessment

def main():
    """Test the health risk assessment"""
    assessor = HealthRiskAssessor()
    
    # Test case 1: Healthy young adult
    profile1 = HealthProfile(
        age=25,
        bmi=22.5,
        smoker=False,
        gender="female",
        exercise_frequency=3
    )
    
    # Test case 2: Older adult with conditions
    profile2 = HealthProfile(
        age=65,
        bmi=31.2,
        smoker=True,
        gender="male",
        exercise_frequency=1,
        chronic_conditions=["diabetes", "hypertension"]
    )
    
    print("\nTest Case 1: Healthy Young Adult")
    print("================================")
    assessment1 = assessor.assess_health_risk(profile1)
    print(f"Risk Score: {assessment1['risk_score']}")
    print(f"Risk Level: {assessment1['risk_level']}")
    print("\nRecommended Services:")
    for service in assessment1['recommended_services']:
        print(f"- {service}")
    print("\nHealth Insights:")
    for insight in assessment1['health_insights']:
        print(f"- {insight}")
        
    print("\nTest Case 2: Older Adult with Conditions")
    print("=======================================")
    assessment2 = assessor.assess_health_risk(profile2)
    print(f"Risk Score: {assessment2['risk_score']}")
    print(f"Risk Level: {assessment2['risk_level']}")
    print("\nRecommended Services:")
    for service in assessment2['recommended_services']:
        print(f"- {service}")
    print("\nHealth Insights:")
    for insight in assessment2['health_insights']:
        print(f"- {insight}")

if __name__ == "__main__":
    main()
