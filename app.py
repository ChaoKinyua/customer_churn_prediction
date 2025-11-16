from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Globals for model/scaler/features. We attempt to load them on startup
# but keep the app import-safe so missing files don't crash the process.
model = None
scaler = None
feature_names = []

def load_resources():
    """Try to load model, scaler and feature names from disk.
    If any resource is missing the app will still start but endpoints
    will return a 503 explaining which resource is unavailable.
    """
    global model, scaler, feature_names
    try:
        model = joblib.load('models/xgboost.pkl')
    except Exception as e:
        warnings.warn(f"Could not load model: {e}")
        model = None

    try:
        scaler = joblib.load('models/scaler.pkl')
    except Exception as e:
        warnings.warn(f"Could not load scaler: {e}")
        scaler = None

    try:
        X_features = pd.read_csv('data/processed/X_features.csv')
        feature_names = X_features.columns.tolist()
    except Exception as e:
        warnings.warn(f"Could not load feature names: {e}")
        feature_names = []

# Load resources on startup
load_resources()

@app.route('/', methods=['GET'])
def home():
    """Health check"""
    return jsonify({
        'status': 'success',
        'message': 'Customer Churn Prediction API v1.0',
        'endpoints': {
            '/predict': 'POST - Predict churn for single customer',
            '/predict-batch': 'POST - Predict churn for multiple customers',
            '/health': 'GET - API health status'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict_single():
    """Predict churn for single customer"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Ensure resources are loaded
        missing = []
        if model is None:
            missing.append('model')
        if scaler is None:
            missing.append('scaler')
        if not feature_names:
            missing.append('feature_names')
        if missing:
            return jsonify({'error': 'Resources unavailable', 'missing': missing}), 503

        # At this point resources should be available; assert to help static analyzers
        assert scaler is not None
        assert model is not None
        
        # Create dataframe with single customer
        customer_data = pd.DataFrame([data])
        
        # Ensure all features are present
        for feat in feature_names:
            if feat not in customer_data.columns:
                customer_data[feat] = 0
        
        # Select only required features in correct order
        customer_data = customer_data[feature_names]
        
        # Scale features
        customer_scaled = scaler.transform(customer_data)

        # Make prediction
        churn_prob = model.predict_proba(customer_scaled)[0][1]
        churn_prediction = model.predict(customer_scaled)[0]
        
        return jsonify({
            'status': 'success',
            'churn_prediction': int(churn_prediction),
            'churn_probability': float(churn_prob),
            'risk_level': 'HIGH' if churn_prob > 0.7 else ('MEDIUM' if churn_prob > 0.4 else 'LOW'),
            'recommendation': 'Immediate retention action needed' if churn_prob > 0.7 else ('Monitor customer' if churn_prob > 0.4 else 'No action needed')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Customer Churn Prediction API...")
    print("Resources loaded - Model:", "✓" if model else "✗")
    print("Resources loaded - Scaler:", "✓" if scaler else "✗")
    print("Resources loaded - Features:", "✓" if feature_names else "✗")
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))