import streamlit as st
import pandas as pd
import pickle
import random

import joblib
joblib.dump(chatgpt, "loan_pred.pkl")
st.header('Loan Eligibility Prediction Using Machine Learning')

c = '''Loan Prediction using Machine Learning Loan Prediction is an essential problem in the banks and finance industries. Accurately prediction whether a loans will be approved or rejected can help financial institutions to manage risk, reduce defaults, and increase profitability.

Algorithms Used:

*Logistic Regression*

*Naive Bayes*

*Support Vector Machine (Linear)*

*K-Nearest Neighbors*

*Decision Tree*

*Random Forest*

*XGBoost*

*Artificial Neural Network (1 Hidden Layer, Keras)*
'''


st.markdown(c)

st.image('https://images.pexels.com/photos/4386321/pexels-photo-4386321.jpeg')

with open("loan_pred.pkl", "rb") as file:
    chatgpt = joblib.load(file)   # Your trained pipeline


#laod data 
url = '''https://github.com/ankitmisk/Heart_Disease_Prediction_ML_Model/blob/main/heart.csv?raw=true'''
df = pd.read_csv(url)
print('Done')

st.sidebar.header('Select features to predict Loan Eligibility')
st.sidebar.image('https://media.tenor.com/v8zmj3VSaPMAAAAM/loan-need-a-loan.gif')

# Example user input (replace with your form inputs)
# Collect input from user via Streamlit widgets
# Example: final_value = pd.DataFrame(user_inputs, index=[0])

# --- FIX: Align user input with training features ---
def preprocess_input(final_value, X_train_columns):
    """
    Ensure user input matches training data columns.
    Extra cols are dropped, missing cols are added as 0.
    """
    return final_value.reindex(columns=X_train_columns, fill_value=0)

# Simulating final_value (replace with your input form)
# Example with dummy values (replace with Streamlit input fields)
final_value = pd.DataFrame({
    "Gender": ["Male"],
    "Married": ["Yes"],
    "Dependents": ["0"],
    "Education": ["Graduate"],
    "Self_Employed": ["No"],
    "ApplicantIncome": [5000],
    "CoapplicantIncome": [0],
    "LoanAmount": [200],
    "Loan_Amount_Term": [360],
    "Credit_History": [1.0],
    "Property_Area": ["Urban"]
    # ⚠ Only include features you used in training
}, index=[0])

for i in df.iloc[:,:-1]:
    min_value, max_value = df[i].agg(['min','max'])

age = st.slider("Select age value", 29, 77, 33)
sex = st.slider("Select sex value", 0, 1, 0)
cp = st.slider("Select cp value", 0, 3, 1)
trestbps = st.slider("Select trestbps value", 94, 200, 94)
chol = st.slider("Select chol value", 126, 564, 302)
fbs = st.slider("Select fbs value", 0, 1, 0)
restecg = st.slider("Select restecg value", 0, 2, 0)

# Get training feature names from pipeline
X_train_columns = chatgpt.feature_names_in_

# Align final_value with training columns
final_value = preprocess_input(final_value, X_train_columns)

# Prediction
ans = chatgpt.predict(final_value)[0]
import time

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting Heart Disease')

place = st.empty()
place.image('https://media.tenor.com/v8zmj3VSaPMAAAAM/loan-need-a-loan.gif',width = 200)

for i in range(100):
    time.sleep(0.05)
    progress_bar.progress(i + 1)

if ans == 0:
    body = f'You are not eligible for loan'
    placeholder.empty()
    place.empty()
    st.success(body)
    progress_bar = st.progress(0)
else:
    body = 'You are eligible for loan'
    placeholder.empty()
    place.empty()
    st.warning(body)
    progress_bar = st.progress(0)

