import requests
import sys

BASE_URL = "http://127.0.0.1:8080"

def check_endpoint(name, url, method="GET", data=None, expected_status=200, expected_content=None):
    print(f"Testing {name} ({url})...", end=" ")
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, data=data)
        
        if response.status_code == expected_status:
            if expected_content and expected_content not in response.text:
                 print(f"FAILED. Status {response.status_code}, but content missing: '{expected_content}'")
                 return False
            print("PASSED")
            return True
        else:
            print(f"FAILED. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def verify():
    results = []
    
    # 1. Login Page
    results.append(check_endpoint("Login Page", f"{BASE_URL}/"))
    
    # 2. Home Page
    results.append(check_endpoint("Home Page", f"{BASE_URL}/home"))
    
    # 3. Yield Page
    results.append(check_endpoint("Yield Page", f"{BASE_URL}/yield"))

    # 4. Recommend Page
    results.append(check_endpoint("Recommend Page", f"{BASE_URL}/recommend"))

    # 5. Recommendation Logic (Punjab)
    # Form data: state=Punjab
    results.append(check_endpoint("Submit Recommendation (State: Punjab)", 
                                  f"{BASE_URL}/get_recommendation", 
                                  method="POST", 
                                  data={"state": "Punjab"}, 
                                  expected_content="Optimal pH"))

    # 6. Recommendation by Soil (Alluvial)
    results.append(check_endpoint("Submit Recommendation (Soil: Alluvial)", 
                                  f"{BASE_URL}/get_recommendation", 
                                  method="POST", 
                                  data={"soil": "Alluvial soil"}, 
                                  expected_content="Alluvial soil"))

    # 6. Yield Prediction Logic
    # Form data matching updated yield.html
    yield_data = {
        "landArea": "5",
        "location": "Punjab",
        "cropType": "Wheat",
        "soilPH": "6.8",
        "season": "Rabi",
        "nitrogen": "120",
        "phosphorus": "60",
        "potassium": "40"
    }
    results.append(check_endpoint("Submit Yield Prediction (Wheat)", 
                                  f"{BASE_URL}/predict_yield", 
                                  method="POST", 
                                  data=yield_data, 
                                  expected_content="Tonnes"))

    # 7. Soil Treatment Advice Verification
    soil_test_data = {
        "landArea": "5",
        "location": "Punjab",
        "cropType": "Rice",
        "soilPH": "8.0",
        "season": "Kharif",
        "nitrogen": "80",
        "phosphorus": "40",
        "potassium": "40"
    }
    results.append(check_endpoint("Submit Soil Advice Test (Rice/High pH)", 
                                  f"{BASE_URL}/predict_yield", 
                                  method="POST", 
                                  data=soil_test_data, 
                                  expected_content="High Alkalinity"))

    if all(results):
        print("\nAll checks PASSED! Application flow verified.")
    else:
        print("\nSome checks FAILED.")

if __name__ == "__main__":
    verify()
