# Example API Requests for Customer Churn Prediction API

## Health Check
```
GET http://127.0.0.1:5000/health
```

## Home/Endpoints
```
GET http://127.0.0.1:5000/
```

## Predict - Single Customer (Minimal Example)
```
POST http://127.0.0.1:5000/predict
Content-Type: application/json

{
  "tenure": 12,
  "MonthlyCharges": 65.00,
  "TotalCharges": 780.00,
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "PhoneService": 1,
  "MultipleLines": 0,
  "PaperlessBilling": 1,
  "InternetService_Fiber optic": 0,
  "InternetService_No": 0,
  "OnlineSecurity_Yes": 1,
  "OnlineBackup_Yes": 0,
  "DeviceProtection_Yes": 0,
  "TechSupport_Yes": 0,
  "StreamingTV_Yes": 0,
  "StreamingMovies_Yes": 0,
  "Contract_One year": 0,
  "Contract_Two year": 0,
  "PaymentMethod_Credit card (automatic)": 0,
  "PaymentMethod_Electronic check": 1,
  "PaymentMethod_Mailed check": 0,
  "CLV": 780.0,
  "MonthlyCost_ratio": 0.083,
  "tenure_years": 1.0,
  "is_high_value": 0,
  "is_new_customer": 0,
  "is_long_term": 0
}
```

Expected Response:
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.25,
  "recommendation": "No action needed",
  "risk_level": "LOW",
  "status": "success"
}
```

## Using curl (PowerShell):
```powershell
$body = @{
    tenure = 12
    MonthlyCharges = 65.00
    TotalCharges = 780.00
    gender = 1
    SeniorCitizen = 0
    Partner = 1
    Dependents = 0
    PhoneService = 1
    MultipleLines = 0
    PaperlessBilling = 1
    InternetService_Fiber_optic = 0
    InternetService_No = 0
    OnlineSecurity_Yes = 1
    OnlineBackup_Yes = 0
    DeviceProtection_Yes = 0
    TechSupport_Yes = 0
    StreamingTV_Yes = 0
    StreamingMovies_Yes = 0
    Contract_One_year = 0
    Contract_Two_year = 0
    PaymentMethod_Credit_card_automatic = 0
    PaymentMethod_Electronic_check = 1
    PaymentMethod_Mailed_check = 0
    CLV = 780.0
    MonthlyCost_ratio = 0.083
    tenure_years = 1.0
    is_high_value = 0
    is_new_customer = 0
    is_long_term = 0
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -Body $body -ContentType "application/json"
```

## Using curl (Bash/Command Line):
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "MonthlyCharges": 65.00,
    "TotalCharges": 780.00,
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "MultipleLines": 0,
    "PaperlessBilling": 1,
    "InternetService_Fiber optic": 0,
    "InternetService_No": 0,
    "OnlineSecurity_Yes": 1,
    "OnlineBackup_Yes": 0,
    "DeviceProtection_Yes": 0,
    "TechSupport_Yes": 0,
    "StreamingTV_Yes": 0,
    "StreamingMovies_Yes": 0,
    "Contract_One year": 0,
    "Contract_Two year": 0,
    "PaymentMethod_Credit card (automatic)": 0,
    "PaymentMethod_Electronic check": 1,
    "PaymentMethod_Mailed check": 0,
    "CLV": 780.0,
    "MonthlyCost_ratio": 0.083,
    "tenure_years": 1.0,
    "is_high_value": 0,
    "is_new_customer": 0,
    "is_long_term": 0
  }'
```

## Response Meanings

- **churn_prediction**: 0 = Will not churn, 1 = Will churn
- **churn_probability**: Likelihood of churn (0.0 to 1.0)
- **risk_level**:
  - LOW: probability < 0.4 (no action needed)
  - MEDIUM: probability 0.4-0.7 (monitor customer)
  - HIGH: probability > 0.7 (immediate retention action needed)
- **recommendation**: Suggested action based on risk level

## Error Responses

### Missing Resources (503)
```json
{
  "error": "Resources unavailable",
  "missing": ["model", "scaler", "feature_names"]
}
```

### Invalid Input (400)
```json
{
  "error": "No data provided"
}
```

### Server Error (500)
```json
{
  "error": "Details of what went wrong"
}
```
