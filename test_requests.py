import requests
import json

# --- API Configuration ---
BASE_URL = 'http://127.0.0.1:5000'
CROP_PREDICT_URL = f'{BASE_URL}/predict'
FERT_URL = f'{BASE_URL}/fertilizer_recommendation'
headers = {'Content-Type': 'application/json'}
# -------------------------

def test_crop_prediction():
    """Tests the crop recommendation endpoint (/predict)."""
    print("--- Testing Crop Recommendation Endpoint (/predict) ---")
    
    # Example input data for Rice/Alluvial soil (from dataset)
    data = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94,
        "soil_type": "Alluvial" # This should match a one-hot feature in model_features.pkl
    }
    
    try:
        response = requests.post(CROP_PREDICT_URL, data=json.dumps(data), headers=headers, timeout=5)
        
        print("Status Code:", response.status_code)
        
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=4))
            assert response.json().get('recommended_crop') == 'rice', "Expected 'rice' but got a different crop."
            print("SUCCESS: Crop prediction seems to be working correctly.")
        else:
            print(f"ERROR: Received non-200 status code. Response: {response.text}")

    except requests.exceptions.ConnectionError:
        print("FATAL ERROR: Could not connect to the Flask API. Ensure 'api_app.py' is running on http://127.0.0.1:5000.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def test_fertilizer_recommendation():
    """Tests the fertilizer recommendation endpoint (/fertilizer_recommendation)."""
    print("\n--- Testing Fertilizer Recommendation Endpoint (/fertilizer_recommendation) ---")
    
    # Test case 1: A crop that is known to exist (e.g., 'rice')
    fert_data = {"crop": "rice"}
    
    try:
        fert_response = requests.post(FERT_URL, data=json.dumps(fert_data), headers=headers, timeout=5)
        
        print(f"Status Code (Rice): {fert_response.status_code}")
        
        if fert_response.status_code == 200:
            print("Response JSON (Rice):")
            print(json.dumps(fert_response.json(), indent=4))
            print("SUCCESS: Fertilizer recommendation for 'rice' retrieved successfully.")
            # Simple check for structure
            assert 'N' in fert_response.json().get('recommended_ratio', {}), "Missing N key in ratio."
        else:
            print(f"ERROR: Received non-200 status code for rice. Response: {fert_response.text}")
            
        # Test case 2: A crop that does not exist
        fert_data_fail = {"crop": "avocado"}
        fert_response_fail = requests.post(FERT_URL, data=json.dumps(fert_data_fail), headers=headers, timeout=5)
        
        print(f"\nStatus Code (Avocado/Fail): {fert_response_fail.status_code}")
        if fert_response_fail.status_code == 404:
            print("SUCCESS: Handled unknown crop ('avocado') correctly with 404 status.")
        else:
            print(f"ERROR: Expected 404 for 'avocado' but got {fert_response_fail.status_code}. Response: {fert_response_fail.text}")

    except requests.exceptions.ConnectionError:
        print("FATAL ERROR: Could not connect to the Flask API. Ensure 'api_app.py' is running on http://127.0.0.1:5000.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    test_crop_prediction()
    test_fertilizer_recommendation()