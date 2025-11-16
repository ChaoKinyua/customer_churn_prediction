"""
Test script for Customer Churn Prediction API
Makes sample requests to test endpoints
"""
import requests
import json
import pandas as pd

BASE_URL = "http://127.0.0.1:5000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_home():
    """Test home endpoint"""
    print("\n" + "="*60)
    print("Testing / (home) endpoint")
    print("="*60)
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_predict_single():
    """Test single prediction with sample customer data"""
    print("\n" + "="*60)
    print("Testing /predict endpoint (Single Customer)")
    print("="*60)
    
    # Sample customer data - adjust based on actual feature names
    customer = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 1,
        "Dependents": 0,
        "tenure": 24,
        "PhoneService": 1,
        "MultipleLines": 0,
        "PaperlessBilling": 1,
        "MonthlyCharges": 79.85,
        "TotalCharges": 1897.80,
        "InternetService_Fiber optic": 0,
        "InternetService_No": 0,
        "OnlineSecurity_No internet service": 0,
        "OnlineSecurity_Yes": 1,
        "OnlineBackup_No internet service": 0,
        "OnlineBackup_Yes": 0,
        "DeviceProtection_No internet service": 0,
        "DeviceProtection_Yes": 0,
        "TechSupport_No internet service": 0,
        "TechSupport_Yes": 0,
        "StreamingTV_No internet service": 0,
        "StreamingTV_Yes": 0,
        "StreamingMovies_No internet service": 0,
        "StreamingMovies_Yes": 0,
        "Contract_One year": 0,
        "Contract_Two year": 0,
        "PaymentMethod_Credit card (automatic)": 0,
        "PaymentMethod_Electronic check": 1,
        "PaymentMethod_Mailed check": 0,
        "CLV": 1916.4,
        "MonthlyCost_ratio": 0.042,
        "tenure_years": 2.0,
        "is_high_value": 0,
        "is_new_customer": 0,
        "is_long_term": 0
    }
    
    print(f"Customer data: {json.dumps(customer, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=customer,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nStatus: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def test_predict_minimal():
    """Test with minimal required data"""
    print("\n" + "="*60)
    print("Testing /predict endpoint (Minimal Data)")
    print("="*60)
    
    # Minimal customer - let API fill missing values with 0
    customer_minimal = {
        "tenure": 12,
        "MonthlyCharges": 65.00,
        "TotalCharges": 780.00
    }
    
    print(f"Minimal customer data: {json.dumps(customer_minimal, indent=2)}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=customer_minimal,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nStatus: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def load_real_sample():
    """Load a real sample from the processed features"""
    print("\n" + "="*60)
    print("Testing /predict endpoint (Real Sample from Dataset)")
    print("="*60)
    
    try:
        X = pd.read_csv('data/processed/X_features.csv')
        # Take first row as sample
        sample = X.iloc[0].to_dict()
        
        print(f"Sample from dataset (first row):")
        print(f"Churn probability estimate based on real features")
        
        response = requests.post(
            f"{BASE_URL}/predict",
            json=sample,
            headers={"Content-Type": "application/json"}
        )
        print(f"\nStatus: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION API - TEST SUITE")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    
    # Run tests
    test_health()
    test_home()
    test_predict_single()
    test_predict_minimal()
    load_real_sample()
    
    print("\n" + "="*60)
    print("Tests Complete!")
    print("="*60)
