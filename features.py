import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data():
    return pd.read_csv('data/processed/telco_cleaned.csv')

def encode_features(df):
    """Encode categorical features"""
    print("Encoding categorical features...")
    
    df_encoded = df.copy()
    # Trim whitespace/newlines from object columns to avoid encoding problems
    obj_cols = df_encoded.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df_encoded[c] = df_encoded[c].astype(str).str.strip()
    
    # Binary features (Yes/No)
    binary_cols = ['PhoneService', 'PaperlessBilling', 'Dependents', 'Partner']
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = (df_encoded[col] == 'Yes').astype(int)
    
    # One-hot encode categorical features
    # Note: dataset column names are lowercase for some fields (e.g. 'gender')
    categorical_cols = ['InternetService', 'OnlineSecurity', 'OnlineBackup', 
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 
                       'StreamingMovies', 'MultipleLines', 'Contract', 'PaymentMethod', 'gender']
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(col, axis=1)
    
    print(f"✓ Features encoded. Shape: {df_encoded.shape}")
    return df_encoded

def create_features(df):
    """Create new features"""
    print("Creating new features...")
    
    df_features = df.copy()
    
    # 1. Customer lifetime value (rough estimate)
    df_features['CLV'] = df_features['MonthlyCharges'] * df_features['tenure']
    
    # 2. Monthly to total charges ratio
    df_features['MonthlyCost_ratio'] = df_features['MonthlyCharges'] / (df_features['TotalCharges'] + 1)
    
    # 3. Tenure in years
    df_features['tenure_years'] = df_features['tenure'] / 12
    
    # 4. High value customer (top 25% by CLV)
    df_features['is_high_value'] = (df_features['CLV'] >= df_features['CLV'].quantile(0.75)).astype(int)
    
    # 5. New customer (less than 6 months)
    df_features['is_new_customer'] = (df_features['tenure'] < 6).astype(int)
    
    # 6. Long-term customer (more than 3 years)
    df_features['is_long_term'] = (df_features['tenure'] > 36).astype(int)
    
    print(f"✓ New features created. Total features: {df_features.shape[1]}")
    return df_features

def prepare_features(df):
    """Prepare all features"""
    print("\n" + "="*50)
    print("FEATURE ENGINEERING")
    print("="*50 + "\n")
    
    # Separate target
    y = df['Churn']
    X = df.drop('Churn', axis=1)
    
    # Drop customer ID (not useful for prediction)
    if 'customerID' in X.columns:
        X = X.drop('customerID', axis=1)
    
    # Encode features
    X = encode_features(X)
    
    # Create new features
    X = create_features(X)
    
    print(f"\nFinal shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Target: {y.shape[0]} samples")
    # y==1 indicates churned in this dataset
    print(f"Class distribution: Churned={y.sum()}, Retained={(y==0).sum()}")
    
    print("\n✓ Feature engineering complete")
    print("="*50 + "\n")
    
    return X, y

if __name__ == "__main__":
    df = load_data()
    X, y = prepare_features(df)
    
    # Save processed data
    X.to_csv('data/processed/X_features.csv', index=False)
    y.to_csv('data/processed/y_target.csv', index=False)
    print("✓ Features saved to data/processed/")