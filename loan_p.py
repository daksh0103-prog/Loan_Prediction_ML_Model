import streamlit as st
import pandas as pd
import pickle

with open("loan_pred.pkl", "wb") as file:
    pickle.dump(model, file)



model_lr = load_model()

# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Loan Eligibility Prediction", layout="wide")

st.image('https://images.pexels.com/photos/8292888/pexels-photo-8292888.jpeg')
st.title("🏦 Loan Eligibility Prediction Using Machine Learning")

st.write("""
Loan Prediction using Machine Learning is an essential problem in the banks and finance industries.  
Accurately predicting whether a loan will be approved or rejected can help financial institutions manage risk, reduce defaults, and increase profitability.
""")

# Sidebar input form
st.sidebar.header("Enter Applicant Details")
st.sidebar.image('https://media1.tenor.com/m/v8zmj3VSaPMAAAAC/loan-need-a-loan.gif')

gender = st.sidebar.selectbox("Gender", ("Male", "Female"))
married = st.sidebar.selectbox("Married", ("Yes", "No"))
dependents = st.sidebar.selectbox("Dependents", ("0", "1", "2", "3+"))
education = st.sidebar.selectbox("Education", ("Graduate", "Not Graduate"))
self_employed = st.sidebar.selectbox("Self Employed", ("Yes", "No"))
applicant_income = st.sidebar.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.sidebar.number_input("Coapplicant Income", min_value=0, step=100)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=10)
loan_amount_term = st.sidebar.number_input("Loan Amount Term (in days)", min_value=0, step=10)
credit_history = st.sidebar.selectbox("Credit History", (1.0, 0.0))
property_area = st.sidebar.selectbox("Property Area", ("Urban", "Semiurban", "Rural"))

# Collect input into dataframe
input_data = {
    "Gender": [gender],
    "Married": [married],
    "Dependents": [dependents],
    "Education": [education],
    "Self_Employed": [self_employed],
    "ApplicantIncome": [applicant_income],
    "CoapplicantIncome": [coapplicant_income],
    "LoanAmount": [loan_amount],
    "Loan_Amount_Term": [loan_amount_term],
    "Credit_History": [credit_history],
    "Property_Area": [property_area]
}

df = pd.DataFrame(input_data)

# Show final input
st.subheader("📊 Final Input Data")
st.write(df)

# Prediction
if st.button("Predict Loan Status"):
    prediction = model.predict(df)[0]
    if prediction == 1:
        st.success("✅ Loan will be APPROVED!")
    else:
        st.error("❌ Loan will be REJECTED!")

# Show algorithms used (static text)
st.markdown("""
---
### Algorithms Used:
- Logistic Regression  
- Naive Bayes  
- Support Vector Machine (Linear)  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- XGBoost  
- Artificial Neural Network (1 Hidden Layer, Keras)  
""")



st.markdown('Designed by : **Daksh Chanana & Dhruv Agnihotri**')
