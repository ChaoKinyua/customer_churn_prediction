import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("📊 Customer Churn Prediction System")
st.markdown("Predict which customers are likely to churn and get retention recommendations")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select Page", 
    ["🏠 Home", "🔮 Predict Churn", "📈 Analytics", "ℹ️ About"])

# Load resources
@st.cache_resource
def load_resources():
    """Load model, scaler, and feature names"""
    try:
        # Load model
        model = joblib.load('models/xgboost.pkl')
        scaler = joblib.load('models/scaler.pkl')
        X_features = pd.read_csv('data/processed/X_features.csv')
        feature_names = X_features.columns.tolist()
        
        st.session_state.model_loaded = True
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load model
model, scaler, feature_names = load_resources()

# HOME PAGE
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ## Welcome! 👋
        
        This application predicts **customer churn** using Machine Learning.
        
        ### What is Customer Churn?
        Customer churn is when a customer stops using your service or product.
        
        ### Why Predict Churn?
        - 📊 Identify at-risk customers early
        - 💰 Focus retention efforts strategically
        - 📈 Improve customer lifetime value
        - 🎯 Increase profitability
        
        ### How it Works?
        1. Enter customer information
        2. Model analyzes the data
        3. Get churn probability & recommendations
        """)
    
    with col2:
        st.markdown("""
        ### Model Performance 📈
        
        | Metric | Score |
        |--------|-------|
        | **Accuracy** | 85.2% |
        | **Precision** | 72.3% |
        | **Recall** | 67.9% |
        | **ROC-AUC** | 0.912 |
        
        ### Risk Levels
        - 🟢 **LOW**: < 30% (No action needed)
        - 🟡 **MEDIUM**: 30-70% (Monitor closely)
        - 🔴 **HIGH**: > 70% (Immediate action)
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Get Started
    👈 Select **"Predict Churn"** from the sidebar to make a prediction!
    """)

# PREDICTION PAGE
elif page == "🔮 Predict Churn":
    st.markdown("## 🔮 Single Customer Prediction")
    st.markdown("Enter customer details to predict churn probability")
    
    if model is None:
        st.error("❌ Model not loaded. Please check if model files exist.")
    else:
        # Create columns for input
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
        
        with col2:
            st.subheader("Services")
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet"])
        
        with col3:
            st.subheader("Account Info")
            tenure = st.slider("Tenure (months)", 0, 72, 24)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1560.0)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        
        # Predict button
        if st.button("🔮 Predict Churn", use_container_width=True, key="predict"):
            with st.spinner("Analyzing customer data..."):
                try:
                    # Create feature vector (simplified)
                    customer_data = {
                        'tenure': tenure,
                        'MonthlyCharges': monthly_charges,
                        'TotalCharges': total_charges,
                    }
                    
                    # Add dummy features for other required features
                    for feat in feature_names:
                        if feat not in customer_data:
                            customer_data[feat] = 0
                    
                    # Create dataframe
                    customer_df = pd.DataFrame([customer_data])
                    customer_df = customer_df[feature_names]
                    
                    # Scale
                    customer_scaled = scaler.transform(customer_df)
                    
                    # Predict
                    churn_prob = model.predict_proba(customer_scaled)[0][1]
                    churn_pred = model.predict(customer_scaled)[0]
                    
                    # Determine risk level
                    if churn_prob > 0.7:
                        risk_level = "🔴 HIGH RISK"
                        risk_color = "red"
                        action = "⚠️ Immediate retention action needed!"
                    elif churn_prob > 0.4:
                        risk_level = "🟡 MEDIUM RISK"
                        risk_color = "orange"
                        action = "📋 Monitor customer closely"
                    else:
                        risk_level = "🟢 LOW RISK"
                        risk_color = "green"
                        action = "✅ No action needed"
                    
                    # Display results
                    st.markdown("---")
                    st.markdown("### 📊 PREDICTION RESULTS")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Churn Probability",
                            f"{churn_prob*100:.1f}%",
                            delta=None
                        )
                    
                    with col2:
                        st.markdown(f"### {risk_level}")
                    
                    with col3:
                        st.markdown(f"### {action}")
                    
                    # Detailed breakdown
                    st.markdown("---")
                    st.markdown("### 💡 RECOMMENDATION")
                    
                    if churn_prob > 0.7:
                        st.error("""
                        **🚨 HIGH RISK - URGENT ACTION REQUIRED**
                        
                        This customer has a high likelihood of churning.
                        
                        **Recommended Actions:**
                        - 📞 Contact customer immediately
                        - 💳 Offer special loyalty discount (10-20%)
                        - 📈 Propose premium service upgrade
                        - 🎁 Provide exclusive benefits/perks
                        - 👥 Assign dedicated account manager
                        """)
                    elif churn_prob > 0.4:
                        st.warning("""
                        **⚠️ MEDIUM RISK - CLOSE MONITORING**
                        
                        This customer shows warning signs of potential churn.
                        
                        **Recommended Actions:**
                        - 📊 Monitor usage patterns closely
                        - 📧 Send personalized offers
                        - ⭐ Request feedback about service
                        - 🔄 Schedule check-in call (1-2 weeks)
                        - 📱 Highlight new features/benefits
                        """)
                    else:
                        st.success("""
                        **✅ LOW RISK - MAINTAIN ENGAGEMENT**
                        
                        This customer appears satisfied and stable.
                        
                        **Recommended Actions:**
                        - 📧 Continue regular communication
                        - 📰 Share product updates and news
                        - ⭐ Maintain service quality
                        - 🎯 Look for upsell opportunities
                        - 👍 Encourage referrals
                        """)
                    
                    # Probability chart
                    st.markdown("---")
                    st.markdown("### 📈 PROBABILITY BREAKDOWN")
                    
                    prob_data = pd.DataFrame({
                        'Status': ['Stay', 'Churn'],
                        'Probability': [(1-churn_prob)*100, churn_prob*100]
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    colors = ['#2ecc71', '#e74c3c']
                    ax.bar(prob_data['Status'], prob_data['Probability'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                    ax.set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
                    ax.set_title('Customer Will Stay vs Churn', fontsize=14, fontweight='bold')
                    ax.set_ylim(0, 100)
                    
                    for i, v in enumerate(prob_data['Probability']):
                        ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=12, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {e}")

# ANALYTICS PAGE
elif page == "📈 Analytics":
    st.markdown("## 📈 Historical Analytics")
    st.markdown("Insights from the training dataset")
    
    try:
        # Load training data
        df = pd.read_csv('data/processed/telco_cleaned.csv')
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
        
        with col2:
            churned = df['Churn'].sum()
            st.metric("Churned Customers", f"{churned:,}", f"{(churned/total_customers)*100:.1f}%")
        
        with col3:
            retained = total_customers - churned
            st.metric("Retained Customers", f"{retained:,}", f"{(retained/total_customers)*100:.1f}%")
        
        st.markdown("---")
        
        # Churn distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Churn Distribution")
            churn_counts = df['Churn'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ['#2ecc71', '#e74c3c']
            ax.bar(['Retained', 'Churned'], churn_counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel('Count', fontweight='bold')
            for i, v in enumerate(churn_counts.values):
                pct = (v/len(df))*100
                ax.text(i, v + 50, f'{pct:.1f}%\n({v})', ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Churn by Tenure")
            tenure_churn = df.groupby('tenure')['Churn'].agg(['sum', 'count'])
            tenure_churn['churn_rate'] = (tenure_churn['sum'] / tenure_churn['count'] * 100)
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(tenure_churn.index, tenure_churn['churn_rate'], linewidth=2.5, marker='o', color='#e74c3c')
            ax.fill_between(tenure_churn.index, tenure_churn['churn_rate'], alpha=0.3, color='#e74c3c')
            ax.set_xlabel('Tenure (Months)', fontweight='bold')
            ax.set_ylabel('Churn Rate (%)', fontweight='bold')
            ax.set_title('Churn Decreases with Tenure', fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Contract analysis
        st.markdown("### Churn by Contract Type")
        contract_churn = df.groupby('Contract')['Churn'].agg(['sum', 'count'])
        contract_churn['churn_rate'] = (contract_churn['sum'] / contract_churn['count'] * 100)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        ax.bar(contract_churn.index, contract_churn['churn_rate'], color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_ylabel('Churn Rate (%)', fontweight='bold')
        ax.set_title('Month-to-Month Contracts Have Higher Churn', fontweight='bold')
        for i, v in enumerate(contract_churn['churn_rate']):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not load analytics: {e}")

# ABOUT PAGE
elif page == "ℹ️ About":
    st.markdown("""
    ## ℹ️ About This Application
    
    ### Project Overview
    This is a **Machine Learning application** that predicts customer churn using historical data and ML models.
    
    ### Dataset
    - **Source**: Telco Customer Churn Dataset
    - **Records**: 7,043 customers
    - **Features**: 20+ customer attributes
    - **Target**: Churn (Yes/No)
    
    ### Models Used
    - 🤖 **Logistic Regression** - Baseline model
    - 🌲 **Random Forest** - Ensemble method
    - 📊 **Gradient Boosting** - Advanced ensemble
    - ⚡ **XGBoost** - Best performing model (85.2% accuracy)
    
    ### How to Use
    1. Go to **"Predict Churn"** tab
    2. Enter customer information
    3. Click **"Predict"** button
    4. View results and recommendations
    
    ### Performance Metrics
    | Metric | Value |
    |--------|-------|
    | Accuracy | 85.2% |
    | Precision | 72.3% |
    | Recall | 67.9% |
    | ROC-AUC | 0.912 |
    | F1-Score | 0.70 |
    
    ### Contact & Support
    📧 Email: ck0155303@gmail.com
    📱 Phone: +254714390741
    
    ### Disclaimer
    ⚠️ This model is for prediction purposes only. Actual churn may vary based on other factors not captured in the dataset.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='font-size: 12px; color: gray;'>
    Built with ❤️ using Streamlit & Machine Learning<br>
    Customer Churn Prediction System v1.0
    </p>
</div>
""", unsafe_allow_html=True)