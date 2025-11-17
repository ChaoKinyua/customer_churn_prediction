# 🎯 Customer Churn Prediction System

A machine learning-powered web application that predicts customer churn and provides actionable retention recommendations. Built with Streamlit for easy deployment and interactive user experience.

**Live Application:** [Customer Churn Prediction System](https://customerchurnprediction-vntzryanyqsot2zuadtfzt.streamlit.app/#services)

---

## 📋 Overview

The Customer Churn Prediction System is an intelligent platform designed to help businesses identify at-risk customers before they leave. Using advanced machine learning algorithms, the system analyzes customer behavior patterns and provides retention strategies to maximize customer lifetime value.

### Key Statistics
- **Total Customers Analyzed:** 7,032
- **Churned Customers:** 1,869 (26.6%)
- **Retained Customers:** 5,163 (73.4%)

---

## 🏠 Home Page

![Home Page Overview](https://via.placeholder.com/800x400?text=Customer+Churn+Prediction+System+Home)

The home page welcomes users and provides:
- System overview and purpose
- Model performance metrics
- Business impact explanation
- Definition of customer churn
- Reasons for churn prediction importance
- Risk level indicators

**Model Performance Metrics:**
| Metric | Score |
|--------|-------|
| **Accuracy** | 85.2% |
| **Precision** | 72.3% |
| **Recall** | 67.9% |
| **ROC-AUC** | 0.912 |

---

## 🔮 Predict Churn Page

![Predict Churn Interface](https://via.placeholder.com/800x500?text=Single+Customer+Prediction+Form)

This interactive page allows you to predict churn probability for individual customers. The prediction form is organized into three intuitive sections:

### Demographics Section
- **Gender:** Select Male or Female
- **Senior Citizen:** Specify if customer is a senior citizen (Yes/No)
- **Partner:** Indicate partner status (Yes/No)

### Services Section
- **Phone Service:** Availability (Yes/No)
- **Internet Service:** Type selection (Fiber Optic, DSL, No)
- **Online Security:** Service subscription status

### Account Info Section
- **Tenure (months):** Slider to set customer relationship duration
- **Monthly Charges ($):** Input field for monthly billing amount
- **Total Charges ($):** Calculated total charges

**How to Use:**
1. Fill in all customer information fields
2. Adjust sliders for tenure and charges
3. Select appropriate service options from dropdowns
4. Click "Predict Churn Probability" button
5. View personalized retention recommendations

---

## 📊 Analytics Page

![Historical Analytics Dashboard](https://via.placeholder.com/800x400?text=Historical+Analytics+Dashboard)

Comprehensive analytics dashboard with insights from the training dataset:

### Key Metrics Displayed
- Total customer count and distribution
- Churn vs. retention breakdown
- Percentage calculations for business insights
- Visual representations of customer segments

**Analytics Features:**
- Customer segmentation analysis
- Churn distribution patterns
- Service utilization trends
- Demographic breakdowns
- Tenure and billing analysis

---

## ℹ️ About Page

Information about the system, including:
- Project description and objectives
- Technical stack and technologies used
- Model architecture and training approach
- Team information and contact details
- FAQ section

---

## 🛠️ Technical Stack

- **Frontend:** Streamlit - Interactive web framework for Python
- **Machine Learning:** Scikit-learn - ML algorithms and model training
- **Data Processing:** Pandas, NumPy - Data manipulation and analysis
- **Visualization:** Plotly, Matplotlib - Interactive charts and graphs
- **Deployment:** Streamlit Cloud - Serverless deployment platform

---

## 🚀 Features

✅ **Real-time Predictions** - Instant churn probability calculations
✅ **Individual Analysis** - Single customer prediction interface
✅ **Historical Analytics** - Comprehensive dataset insights
✅ **Retention Recommendations** - AI-generated retention strategies
✅ **User-Friendly Interface** - Intuitive navigation and forms
✅ **Mobile Responsive** - Works seamlessly on all devices
✅ **Model Performance Metrics** - Transparent accuracy and performance data
✅ **Interactive Dashboard** - Visual analytics and data exploration

---

## 📈 Model Performance

The churn prediction model achieves strong performance metrics:

- **Accuracy:** 85.2% - Overall correctness of predictions
- **Precision:** 72.3% - Accuracy of positive predictions
- **Recall:** 67.9% - Ability to identify actual churners
- **ROC-AUC:** 0.912 - Excellent discrimination between churners and retainers

---

## 🎯 Use Cases

1. **Proactive Retention:** Identify high-risk customers before they leave
2. **Resource Allocation:** Target retention efforts on customers most likely to churn
3. **Revenue Protection:** Minimize revenue loss through targeted interventions
4. **Customer Insights:** Understand patterns and factors driving churn
5. **Campaign Optimization:** Personalize retention strategies by customer segment

---

## 📱 Navigation Guide

| Page | Purpose | Key Features |
|------|---------|--------------|
| **Home** | System introduction | Model metrics, churn definition, business impact |
| **Predict Churn** | Individual predictions | Customer form, probability output, recommendations |
| **Analytics** | Dataset insights | Historical trends, customer statistics, patterns |
| **About** | System information | Project details, tech stack, contact info |

---

## 🔒 Data Privacy

- All predictions are processed in real-time
- No personal customer data is stored
- Predictions are based on behavioral patterns only
- Compliant with data protection standards

---

## 💡 Getting Started

1. Visit the live application link
2. Navigate to "Predict Churn" from the sidebar
3. Enter customer information in the form
4. Review churn probability and recommendations
5. Explore "Analytics" for historical insights
6. Check "About" for technical details

---

## 📞 Support & Contact

For questions, issues, or feature requests, please refer to the About page for contact information or open an issue in the project repository.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset source: Telecom Customer Churn Data
- Built with Streamlit
- Powered by machine learning and data science

---

**Last Updated:** November 2025
**Version:** 1.0.0
