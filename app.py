import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and thresholds
model = joblib.load("best_churn_pipeline.pkl")
balance_threshold = joblib.load("balance_threshold.pkl")
balance_median = joblib.load("balance_median.pkl")

st.title("Customer Churn Prediction App")

st.header("Enter Customer Details")

credit_score = st.number_input("Credit Score", 300, 850, 650)
country = st.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 90, 40)
tenure = st.slider("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
products_number = st.selectbox("Number of Products", [1,2,3,4])
credit_card = st.selectbox("Has Credit Card", [0,1])
active_member = st.selectbox("Is Active Member", [0,1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)

if st.button("Predict Churn"):

    data = pd.DataFrame([{
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
    }])

    # Feature engineering
    data['balance_per_product'] = data['balance'] / max(products_number,1)
    data['salary_balance_ratio'] = data['estimated_salary'] / max(balance,1)
    
    bins = [0,25,35,45,55,65,100]
    labels = ['<25','25-34','35-44','45-54','55-64','65+']
    data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels)
    
    data['tenure_bucket'] = pd.cut(
        data['tenure'],
        bins=[-1,0,2,5,10,100],
        labels=['0','1-2','3-5','6-10','10+']
    )
    
    data['high_balance'] = (balance > balance_threshold).astype(int)
    
    data['engagement_score'] = (
        active_member +
        int(products_number > 1) +
        int(balance > balance_median)
    )
    
    data['activity_intensity'] = tenure * products_number
    
    data['basic_risk_score'] = (
        int(tenure <= 2) +
        int(products_number == 1) +
        int(active_member == 0)
    )

    # Predict
    prob = model.predict_proba(data)[0,1]
    
    if prob > 0.7:
        risk = "High Risk"
    elif prob > 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {prob:.2%}")
    st.write(f"Risk Level: {risk}")
    
    if risk == "High Risk":
        st.error("Immediate retention action recommended.")
    elif risk == "Medium Risk":
        st.warning("Monitor and offer engagement incentives.")
    else:
        st.success("Customer likely to remain active.")
