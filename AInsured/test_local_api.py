import requests
import json

def test_local_api():
    """Test the AInsued API running locally"""
    base_url = "http://127.0.0.1:8000"
    
    print("\nTesting AInsued API")
    print("==================")
    
    # Test 1: Basic API access
    print("\n1. Testing API access...")
    response = requests.get(base_url)
    if response.ok:
        print("✓ API is accessible")
        print("Response:", json.dumps(response.json(), indent=2))
    else:
        print("✗ Could not access API")
        return
        
    # Test 2: Get available categories
    print("\n2. Getting available categories...")
    response = requests.get(f"{base_url}/categories")
    if response.ok:
        print("✓ Categories retrieved")
        categories = response.json()
        
        print("\nAvailable Provider Types:", len(categories['physician']['provider_types']))
        print("Sample:", categories['physician']['provider_types'][:3])
        
        print("\nAvailable Procedure Codes:", len(categories['physician']['procedure_codes']))
        print("Sample:", categories['physician']['procedure_codes'][:3])
        
        print("\nAvailable DRG Codes:", len(categories['hospital']['drg_codes']))
        print("Sample:", categories['hospital']['drg_codes'][:3])
    else:
        print("✗ Could not get categories")
        return
        
    # Test 3: Make a physician cost prediction
    print("\n3. Testing physician cost prediction...")
    physician_request = {
        "provider_type": categories['physician']['provider_types'][0],  # Internal Medicine
        "procedure_code": categories['physician']['procedure_codes'][0],  # 99217
        "place_of_service": categories['physician']['places_of_service'][0],
        "num_beneficiaries": 100,
        "num_services": 150,
        "submitted_charge": 200.0,
        "allowed_amount": 150.0
    }
    
    print("\nRequest data:")
    print(json.dumps(physician_request, indent=2))
    
    response = requests.post(f"{base_url}/predict/physician", json=physician_request)
    if response.ok:
        print("\nResponse:")
        result = response.json()
        print(f"Predicted Cost: ${result['predicted_cost']:,.2f}")
        print(f"Confidence Score: {result['confidence_score']:.2%}")
        print("\nFactors affecting prediction:")
        for factor in result['factors']:
            print(f"- {factor['name']}: {factor['importance']:.2%}")
    else:
        print("✗ Prediction failed:", response.text)
        
    # Test 4: Make a hospital cost prediction
    print("\n4. Testing hospital cost prediction...")
    hospital_request = {
        "drg_code": str(categories['hospital']['drg_codes'][0]),  # Convert DRG code to string
        "num_discharges": 50,
        "submitted_charge": 25000.0
    }
    
    print("\nRequest data:")
    print(json.dumps(hospital_request, indent=2))
    
    response = requests.post(f"{base_url}/predict/hospital", json=hospital_request)
    if response.ok:
        print("\nResponse:")
        result = response.json()
        print(f"Predicted Cost: ${result['predicted_cost']:,.2f}")
        print(f"Confidence Score: {result['confidence_score']:.2%}")
        print("\nFactors affecting prediction:")
        for factor in result['factors']:
            print(f"- {factor['name']}: {factor['importance']:.2%}")
    else:
        print("✗ Prediction failed:", response.text)

if __name__ == "__main__":
    test_local_api()
