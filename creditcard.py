import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

@st.cache_resource
def train_models():
    creditcard_df = pd.read_csv(r'C:\Users\Abhinn\Desktop\ML PROJECTS\creditcard.csv')

    X = creditcard_df.drop('Class', axis=1)
    y = creditcard_df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2
    )

    sm = SMOTE(random_state=2)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_bal = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=2, n_jobs=-1)
    rf_model.fit(X_train_bal, y_train_bal)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)

    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              max_depth=6, n_estimators=200, random_state=2, n_jobs=-1)
    xgb_model.fit(X_train_bal, y_train_bal)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_pred)

    return rf_model, xgb_model, scaler, rf_acc, xgb_acc, X.columns.tolist()

rf_model, xgb_model, scaler, rf_acc, xgb_acc, feature_columns = train_models()

st.title("💳 Credit Card Fraud Detection App")
st.write("This app uses the **Random Forest** and **XGBoost** models trained in our Jupyter Notebook.")

st.subheader("Model Performance on Test Data")
col1, col2 = st.columns(2)
col1.metric("Random Forest Accuracy", f"{rf_acc:.4f}")
col2.metric("XGBoost Accuracy", f"{xgb_acc:.4f}")

st.markdown("---")
st.subheader("Enter Transaction Details for Prediction")
user_input = []

for col in feature_columns:
    val = st.number_input(f"{col}:", value=0.0, format="%.6f")
    user_input.append(val)

model_choice = st.selectbox("Choose Model", ("Random Forest", "XGBoost"))

if st.button("Predict"):
    features_array = np.array(user_input).reshape(1, -1)
    features_scaled = scaler.transform(features_array)

    if model_choice == "Random Forest":
        prediction = rf_model.predict(features_scaled)[0]
    else:
        prediction = xgb_model.predict(features_scaled)[0]

    if prediction == 0:
        st.success("✅ Legitimate Transaction")
    else:
        st.error("⚠️ Fraudulent Transaction Detected")