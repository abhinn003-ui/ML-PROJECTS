# 🤖 ML Projects – Credit Card Fraud Detection & Loan Approval Prediction

A collection of **Python-based machine learning projects** focused on real-world financial decision support.  
This repository includes two major projects:  

1. **Credit Card Fraud Detection** – Detecting fraudulent transactions using machine learning with imbalanced datasets.  
2. **Loan Approval Prediction** – Predicting loan eligibility based on applicant details, with multiple ML models and feature engineering.  

---

## 🔍 Project 1: Credit Card Fraud Detection
- **Goal:** Identify fraudulent credit card transactions in highly imbalanced datasets.  
- **Tech Stack:**  
  - `numpy`, `pandas`, `streamlit`  
  - `scikit-learn` (train_test_split, RandomForestClassifier, StandardScaler, metrics)  
  - `xgboost` (XGBClassifier)  
  - `imblearn` (SMOTE for oversampling)  
- **Features:**  
  - Data preprocessing & standardization  
  - Handling class imbalance with SMOTE  
  - Model training using Random Forest and XGBoost  
  - Evaluation with accuracy score & classification report  
  - Streamlit-based interface for interactive prediction  

---

## 🔍 Project 2: Loan Approval Predictor
- **Goal:** Predict whether a loan should be approved based on applicant information.  
- **Tech Stack:**  
  - Python, Scikit-learn, Streamlit  
- **Models Used:**  
  - KNN, Random Forest, SVM, Logistic Regression  
- **Highlights:**  
  - Achieved **83% accuracy** with Random Forest  
  - Improved recall by **10%** and precision by **15%** through feature engineering  
  - Interactive **dashboard for performance tracking** and threshold tuning  

---

## 📊 Outcomes
- Hands-on application of **data preprocessing, feature engineering, model building, and evaluation**.  
- Experience with **imbalanced datasets** (fraud detection) and **classification metrics optimization** (loan approval).  
- Real-world aligned projects that demonstrate **machine learning for financial decision-making**.  

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-projects.git
cd ml-projects

# Install dependencies
pip install -r requirements.txt
