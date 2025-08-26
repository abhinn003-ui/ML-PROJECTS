# 🤖 Machine Learning Projects – Credit Card Fraud Detection & Loan Approval Prediction

This repository contains two real-world **machine learning projects** that demonstrate end-to-end data science workflows for financial decision-making:  

1. **Credit Card Fraud Detection** – A system for detecting fraudulent transactions in highly imbalanced datasets.  
2. **Loan Approval Predictor** – A model that predicts loan eligibility based on applicant profiles using multiple ML algorithms.  

These projects cover **data preprocessing, feature engineering, model training, evaluation, and deployment with interactive dashboards.**

---

## 🔍 Project 1: Credit Card Fraud Detection

### 📌 Problem Statement
Credit card companies face the challenge of identifying fraudulent transactions among millions of daily records. The dataset is **highly imbalanced**, with very few fraud cases compared to normal transactions, making detection more complex.  

### ⚙️ Technologies Used
- **Core Libraries:** `numpy`, `pandas`, `streamlit`  
- **ML Frameworks:** `scikit-learn`, `xgboost`  
- **Preprocessing:** `StandardScaler`  
- **Imbalanced Handling:** `SMOTE` (Synthetic Minority Oversampling) from `imblearn`  
- **Evaluation Metrics:** `accuracy_score`, `classification_report`  

### 🧠 Workflow
1. Data preprocessing and normalization with **StandardScaler**  
2. Handling class imbalance using **SMOTE**  
3. Training with **Random Forest** and **XGBoost** classifiers  
4. Evaluating models with accuracy, precision, recall, and F1-score  
5. Building a **Streamlit web app** for interactive fraud detection  

### ✅ Key Outcomes
- Achieved **high recall** for fraud detection (critical to minimize missed fraud cases)  
- Built a lightweight **Streamlit dashboard** for live predictions  
- Gained experience in **imbalanced dataset handling** and **ensemble learning**  

---

## 🔍 Project 2: Loan Approval Predictor

### 📌 Problem Statement
Financial institutions need accurate models to evaluate whether a loan application should be approved. This project builds ML models that predict loan eligibility based on factors like income, credit history, and loan amount.  

### ⚙️ Technologies Used
- **Core Libraries:** Python, `pandas`, `numpy`, `streamlit`  
- **ML Frameworks:** `scikit-learn` (KNN, Random Forest, SVM, Logistic Regression)  

### 🧠 Workflow
1. Data preprocessing and cleaning of applicant information  
2. Feature engineering to improve prediction accuracy  
3. Training multiple ML models (KNN, Random Forest, SVM, Logistic Regression)  
4. Evaluating models with accuracy, precision, recall, and confusion matrices  
5. Deploying via **Streamlit dashboard** with threshold tuning  

### ✅ Key Outcomes
- Achieved **83% accuracy** using Random Forest  
- Improved **recall by 10%** and **precision by 15%** via feature engineering  
- Built an interactive **Streamlit dashboard** for testing different thresholds and monitoring model performance  

---

## 📊 Comparative Outcomes
- **Fraud Detection:** Focused on **recall** to minimize undetected frauds  
- **Loan Predictor:** Balanced **accuracy, precision, and recall** for fair decisions  
- Both projects apply **real-world financial datasets** and demonstrate skills in **model selection, evaluation, and deployment**  

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ml-projects.git
cd ml-projects

# Install dependencies
pip install -r requirements.txt
