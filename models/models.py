import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_features():
    """Load features and target"""
    X = pd.read_csv('data/processed/X_features.csv')
    y = pd.read_csv('data/processed/y_target.csv').values.ravel()
    return X, y

def split_data(X, y):
    """Split data with stratification"""
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def handle_imbalance(X_train, y_train):
    """Handle class imbalance using SMOTE"""
    print("Handling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    res = smote.fit_resample(X_train, y_train)
    # fit_resample may return (X_res, y_res) or (X_res, y_res, sample_weight)
    X_train_balanced, y_train_balanced = res[0], res[1]
    print(f"✓ Balanced: {X_train_balanced.shape[0]} samples")
    return X_train_balanced, y_train_balanced

def scale_features(X_train, X_test):
    """Standardize features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models/scaler.pkl')
    return X_train_scaled, X_test_scaled

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models"""
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    results = {}
    
    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['Logistic Regression'] = evaluate_model(lr, X_test, y_test, y_pred_lr)
    joblib.dump(lr, 'models/logistic_regression.pkl')
    
    # 2. Random Forest
    print("\n2. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['Random Forest'] = evaluate_model(rf, X_test, y_test, y_pred_rf)
    joblib.dump(rf, 'models/random_forest.pkl')
    
    # 3. Gradient Boosting
    print("\n3. Training Gradient Boosting...")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    results['Gradient Boosting'] = evaluate_model(gb, X_test, y_test, y_pred_gb)
    joblib.dump(gb, 'models/gradient_boosting.pkl')
    
    # 4. XGBoost
    print("\n4. Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test, y_pred_xgb)
    joblib.dump(xgb_model, 'models/xgboost.pkl')
    
    return results, {'lr': lr, 'rf': rf, 'gb': gb, 'xgb': xgb_model}

def evaluate_model(model, X_test, y_test, y_pred):
    """Evaluate model performance"""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    metrics = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'ROC-AUC': auc
    }
    
    print(f"✓ Accuracy: {acc:.4f}")
    print(f"✓ Precision: {prec:.4f}")
    print(f"✓ Recall: {rec:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")
    print(f"✓ ROC-AUC: {auc:.4f}")
    
    return metrics

def plot_model_comparison(results):
    """Plot model comparison"""
    df_results = pd.DataFrame(results).T
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Metrics comparison
    df_results.plot(kind='bar', ax=axes[0], alpha=0.8)
    axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ROC-AUC comparison
    df_results['ROC-AUC'].plot(kind='barh', ax=axes[1], color='#3498db', alpha=0.8)
    axes[1].set_title('ROC-AUC Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('ROC-AUC Score', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('reports/05_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: 05_model_comparison.png")

def main():
    """Main training pipeline"""
    print("="*50)
    print("CUSTOMER CHURN - MODEL TRAINING")
    print("="*50)
    
    # Load data
    X, y = load_features()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Handle imbalance
    X_train, y_train = handle_imbalance(X_train, y_train)
    
    # Scale features
    X_train, X_test = scale_features(X_train, X_test)
    
    # Train models
    results, models = train_models(X_train, X_test, y_train, y_test)
    
    # Plot comparison
    plot_model_comparison(results)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())
    
    # Save best model
    best_model = max(results, key=lambda x: results[x]['ROC-AUC'])
    print(f"\n✓ Best Model: {best_model} (ROC-AUC: {results[best_model]['ROC-AUC']:.4f})")
    print("✓ All models saved to models/")

if __name__ == "__main__":
    main()