import streamlit as st
import pandas as pd
import joblib
 
st.title("Employee Attrition Prediction App")

pipeline = joblib.load("employee_attrition.pkl")
columns = joblib.load("columns.pkl")  # Load saved column names

# User Inputs
age = st.number_input("Age", 18, 60, 30)
monthly_income = st.number_input("Monthly Income", 1000, 30000, 5000)
gender = st.selectbox("Gender", ["Male", "Female"])
job_role = st.selectbox("Job Role", ["Research Scientist", "Sales Executive", "Manager"])

# Create template with all columns
input_df = pd.DataFrame(columns=columns)
input_df.loc[0] = [None] * len(columns)

# Fill only the features you collect
input_df.loc[0, "Age"] = age
input_df.loc[0, "MonthlyIncome"] = monthly_income
input_df.loc[0, "Gender"] = gender
input_df.loc[0, "JobRole"] = job_role

# Prediction
if st.button("Predict"):
    pred = pipeline.predict(input_df)[0]
    st.write("Attrition:", "Yes" if pred == 1 else "No")
