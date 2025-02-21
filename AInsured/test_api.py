import requests
import json

def test_prediction():
    """Test the prediction API endpoint"""
    url = "http://127.0.0.1:8000/predict"
    
    # Sample health information
    health_info = {
        "age": 35,
        "state": "CA",
        "smoker": False,
        "bmi": 24.5,
        "exercise_frequency": 3,
        "chronic_conditions": []
    }
    
    try:
        # Make POST request to the API
        response = requests.post(url, json=health_info)
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction Results:")
            print("-------------------")
            print(f"Expected Annual Cost: ${result['expected_annual_cost']:,.2f}")
            print("\nRecommended Plans:")
            for plan in result['recommended_plans']:
                print(f"\n- {plan['name']}")
                print(f"  Premium: ${plan['premium']:,.2f}")
                print(f"  Deductible: ${plan['deductible']:,.2f}")
                print(f"  Out of Pocket Max: ${plan['out_of_pocket_max']:,.2f}")
                print(f"  Coverage Level: {plan['coverage_level']}")
            print(f"\nConfidence Score: {result['confidence_score']:.2%}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the API server is running (uvicorn main:app --reload)")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_prediction()
