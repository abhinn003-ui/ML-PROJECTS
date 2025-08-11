import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

data = pd.read_csv(r'C:\Users\Abhinn\Desktop\ML PROJECTS\LoanApprovalPrediction.csv')
data.drop(['Loan_ID'],axis=1,inplace=True)

gendermap = {'Male':1,'Female':0}
marriedmap = {'Yes':1,'No':0}
educationmap = {'Graduate':1,'Not Graduate':0}
selfemployedmap = {'Yes':1,'No':0}
propertyareamap = {'Urban':2,'Semiurban':1,'Rural':0}

labelencoder = preprocessing.LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = labelencoder.fit_transform(data[col])

for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

X = data.drop(['Loan_Status'], axis=1)
Y = data['Loan_Status']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.4, random_state=1)

rfc = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
rfc.fit(Xtrain, Ytrain)

st.markdown(
    """
    <style>
     header {visibility: hidden;}
    .stApp {
        background-color: #E9967A;
    }
     h1 {
        color: #000000;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Loan Eligibility Predictor")
st.write("Fill out the details below to check loan eligibility.")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
SelfEmployed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income(monthly)", min_value=0.0, step=100.0)
CoapplicantIncome = st.number_input("Coapplicant Income(monthly)", min_value=0.0, step=100.0)
LoanAmount = st.number_input("Loan Amount(in thousands)", min_value=0.0, step=1.0)
LoanAmountTerm = st.number_input("Loan Amount Term (in months)", min_value=0.0, step=1.0)
CreditHistory = st.selectbox("Credit History", ["1", "0"])
PropertyArea = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Check Eligibility"):
    try:
        Gender = gendermap[Gender]
        Married = marriedmap[Married]
        Dependents = int(Dependents.replace("3+", "3"))
        Education = educationmap[Education]
        SelfEmployed = selfemployedmap[SelfEmployed]
        CreditHistory = int(CreditHistory)
        PropertyArea = propertyareamap[PropertyArea]

        inputs = pd.DataFrame({
            'Gender': [Gender],
            'Married': [Married],
            'Dependents': [Dependents],
            'Education': [Education],
            'Self_Employed': [SelfEmployed],
            'ApplicantIncome': [ApplicantIncome],
            'CoapplicantIncome': [CoapplicantIncome],
            'LoanAmount': [LoanAmount],
            'Loan_Amount_Term': [LoanAmountTerm],
            'Credit_History': [CreditHistory],
            'Property_Area': [PropertyArea]
        })

        prediction = rfc.predict(inputs)[0]
        status = "Eligible" if prediction == 1 else "Not Eligible"

        st.success(f"The individual is **{status}** for the loan.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
