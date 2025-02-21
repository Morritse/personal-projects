import requests
import json

def test_health_assessment():
    """Test the health and cost assessment API"""
    base_url = "http://127.0.0.1:8000"
    
    print("\nTesting Health & Cost Assessment API")
    print("===================================")
    
    # Test 1: Get available conditions
    print("\n1. Getting available conditions...")
    response = requests.get(f"{base_url}/conditions")
    if response.ok:
        print("✓ Conditions retrieved")
        conditions = response.json()
        print("\nSupported conditions:")
        for condition in conditions['conditions']:
            print(f"- {condition}")
    
    # Test 2: Get preventive services
    print("\n2. Getting preventive services...")
    response = requests.get(f"{base_url}/preventive-services")
    if response.ok:
        print("✓ Services retrieved")
        services = response.json()
        print("\nPreventive services:")
        print("- General:", ", ".join(services['all'][:3]) + "...")
        print("- Female 40+:", ", ".join(services['female']['40+']))
        print("- Male 50+:", ", ".join(services['male']['50+']))
    
    # Test 3: Assess healthy young adult
    print("\n3. Testing assessment for healthy young adult...")
    young_profile = {
        "age": 25,
        "bmi": 22.5,
        "smoker": False,
        "gender": "female",
        "exercise_frequency": 3,
        "chronic_conditions": None,
        "family_history": None
    }
    
    print("\nRequest data:")
    print(json.dumps(young_profile, indent=2))
    
    response = requests.post(f"{base_url}/assess", json=young_profile)
    if response.ok:
        result = response.json()
        print("\nAssessment Results:")
        print(f"Risk Level: {result['risk_assessment']['risk_level']}")
        print(f"Risk Score: {result['risk_assessment']['risk_score']}")
        print("\nEstimated Annual Costs:")
        print(f"Preventive Care: ${result['estimated_costs']['preventive_care']:,.2f}")
        print(f"Total Annual: ${result['estimated_costs']['estimated_annual']:,.2f}")
        print("\nRecommended Services:")
        for service in result['recommended_services'][:5]:
            print(f"- {service}")
    
    # Test 4: Assess older adult with conditions
    print("\n4. Testing assessment for older adult with conditions...")
    older_profile = {
        "age": 65,
        "bmi": 31.2,
        "smoker": True,
        "gender": "male",
        "exercise_frequency": 1,
        "chronic_conditions": ["diabetes", "hypertension"],
        "family_history": ["heart_disease"]
    }
    
    print("\nRequest data:")
    print(json.dumps(older_profile, indent=2))
    
    response = requests.post(f"{base_url}/assess", json=older_profile)
    if response.ok:
        result = response.json()
        print("\nAssessment Results:")
        print(f"Risk Level: {result['risk_assessment']['risk_level']}")
        print(f"Risk Score: {result['risk_assessment']['risk_score']}")
        print("\nEstimated Annual Costs:")
        print(f"Preventive Care: ${result['estimated_costs']['preventive_care']:,.2f}")
        print(f"Total Annual: ${result['estimated_costs']['estimated_annual']:,.2f}")
        print("\nRecommended Services:")
        for service in result['recommended_services'][:5]:
            print(f"- {service}")
        print("\nPreventive Measures:")
        for measure in result['preventive_measures']:
            print(f"- {measure}")

if __name__ == "__main__":
    test_health_assessment()
