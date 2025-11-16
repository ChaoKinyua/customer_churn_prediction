import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data():
    path = 'data/processed/telco_cleaned.csv'
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at '{path}'.\nRun the data cleaning step first: `python data_loader.py` or use the debug loader.`"
        )
    return pd.read_csv(path)

def plot_churn_distribution(df):
    """Plot churn distribution"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # Ensure counts are ordered [0, 1] -> [Retained, Churned]
    churn_counts = df['Churn'].value_counts().reindex([0, 1], fill_value=0)
    colors = ['#2ecc71', '#e74c3c']  # Green for retained, Red for churned

    ax.bar(['Retained', 'Churned'], churn_counts.values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
    
    # Add percentages
    for i, v in enumerate(churn_counts.values):
        percentage = (v / len(df)) * 100
        ax.text(i, v + 50, f'{percentage:.1f}%\n({v})', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/01_churn_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 01_churn_distribution.png")

def plot_tenure_churn(df):
    """Plot tenure vs churn"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Group by tenure and calculate churn rate
    tenure_churn = df.groupby('tenure')['Churn'].agg(['sum', 'count'])
    tenure_churn['churn_rate'] = (tenure_churn['sum'] / tenure_churn['count'] * 100)
    
    ax.plot(tenure_churn.index, tenure_churn['churn_rate'], linewidth=2, marker='o', color='#e74c3c')
    ax.fill_between(tenure_churn.index, tenure_churn['churn_rate'], alpha=0.3, color='#e74c3c')
    
    ax.set_xlabel('Tenure (Months)', fontsize=12)
    ax.set_ylabel('Churn Rate (%)', fontsize=12)
    ax.set_title('Churn Rate by Customer Tenure', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/02_tenure_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 02_tenure_churn.png")

def plot_monthly_charges_churn(df):
    """Plot monthly charges vs churn"""
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Distribution by churn status
    df[df['Churn'] == 0]['MonthlyCharges'].hist(bins=30, alpha=0.7, label='Retained', ax=ax[0], color='#2ecc71')
    df[df['Churn'] == 1]['MonthlyCharges'].hist(bins=30, alpha=0.7, label='Churned', ax=ax[0], color='#e74c3c')
    ax[0].set_xlabel('Monthly Charges ($)', fontsize=12)
    ax[0].set_ylabel('Frequency', fontsize=12)
    ax[0].set_title('Monthly Charges Distribution', fontsize=14, fontweight='bold')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # Box plot
    df.boxplot(column='MonthlyCharges', by='Churn', ax=ax[1])
    ax[1].set_xlabel('Churn Status (0=Retained, 1=Churned)', fontsize=12)
    ax[1].set_ylabel('Monthly Charges ($)', fontsize=12)
    ax[1].set_title('Monthly Charges by Churn Status', fontsize=14, fontweight='bold')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig('reports/03_charges_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 03_charges_churn.png")

def plot_contract_churn(df):
    """Plot contract type vs churn"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    contract_churn = df.groupby('Contract')['Churn'].agg(['sum', 'count'])
    contract_churn['churn_rate'] = (contract_churn['sum'] / contract_churn['count'] * 100)
    
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax.bar(contract_churn.index, contract_churn['churn_rate'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Churn Rate (%)', fontsize=12)
    ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(contract_churn['churn_rate']):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('reports/04_contract_churn.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 04_contract_churn.png")

def generate_report():
    """Generate full EDA report"""
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50 + "\n")
    # Ensure output folder exists
    os.makedirs('reports', exist_ok=True)

    df = load_data()
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}\n")
    
    # Generate plots
    plot_churn_distribution(df)
    plot_tenure_churn(df)
    plot_monthly_charges_churn(df)
    plot_contract_churn(df)
    
    print("\n✓ All EDA plots saved to reports/")
    print("="*50 + "\n")

if __name__ == "__main__":
    generate_report()