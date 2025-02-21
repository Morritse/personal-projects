from models.predictor import HealthCostPredictor
from pathlib import Path
import sys

def test_predictor():
    """Test the HealthCostPredictor class"""
    predictor = HealthCostPredictor()
    
    # Test health information
    test_health_info = {
        "age": 35,
        "state": "CA",
        "smoker": False,
        "bmi": 24.5,
        "exercise_frequency": 3,
        "chronic_conditions": []
    }
    
    # Get prediction
    prediction = predictor.get_prediction(test_health_info)
    
    print("\nTest Results:")
    print("-------------")
    print(f"Expected Annual Cost: ${prediction['expected_annual_cost']:,.2f}")
    print("\nRecommended Plans:")
    for plan in prediction['recommended_plans']:
        print(f"\n- {plan['name']}")
        print(f"  Premium: ${plan['premium']:,.2f}")
        print(f"  Deductible: ${plan['deductible']:,.2f}")
        print(f"  Out of Pocket Max: ${plan['out_of_pocket_max']:,.2f}")
        print(f"  Coverage Level: {plan['coverage_level']}")
    print(f"\nConfidence Score: {prediction['confidence_score']:.2%}")

def main():
    """Main function to test the initial setup"""
    print("\nTesting AInsued Initial Setup")
    print("============================")
    
    # Check project structure
    required_dirs = ['data', 'models', 'api']
    project_root = Path(__file__).parent
    
    print("\nChecking project structure...")
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ directory found")
        else:
            print(f"✗ {dir_name}/ directory missing")
    
    # Test predictor
    print("\nTesting prediction model...")
    try:
        test_predictor()
        print("\n✓ Predictor test completed successfully")
    except Exception as e:
        print(f"\n✗ Predictor test failed: {e}")
    
    print("\nSetup test completed.")

if __name__ == "__main__":
    main()
