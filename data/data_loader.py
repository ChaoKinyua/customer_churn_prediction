import pandas as pd
import numpy as np
import os

def load_data():
    """Load raw customer churn data"""
    file_path = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data not found at {file_path}")  
    df = pd.read_csv(file_path)
    print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def clean_data(df):
    """Clean and preprocess data"""
    print("Cleaning data...")
    
    # Remove spaces from column names
    df.columns = df.columns.str.strip()
    
    # Handle 'Total Charges' - convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Remove rows with missing TotalCharges
    df = df.dropna(subset=['TotalCharges'])
    
    # Convert 'Churn' to binary (1 = Yes, 0 = No)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    print(f"✓ Data cleaned: {df.shape[0]} rows remaining")
    return df

def basic_stats(df):
    """Print basic statistics"""
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    print(f"Total Customers: {len(df)}")
    print(f"Churn Count: {df['Churn'].sum()} ({df['Churn'].mean()*100:.1f}%)")
    print(f"Retained: {(1-df['Churn']).sum()} ({(1-df['Churn']).mean()*100:.1f}%)")
    print(f"\nColumns: {df.shape[1]}")
    print(f"Data types:\n{df.dtypes}")
    print("="*50 + "\n")

# Usage
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    basic_stats(df)
    
    # Save cleaned data
    df.to_csv('data/processed/telco_cleaned.csv', index=False)
    print("✓ Cleaned data saved")