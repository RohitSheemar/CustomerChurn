import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the pre-trained model, scaler, and encoders
model = joblib.load('random_forest_churn_model_reduced.pkl')
scaler = joblib.load('scaler_reduced.pkl')
encoders = joblib.load('encoders.pkl')

# Streamlit app title
st.title('Telecom Customer Churn Prediction')

# Input features
st.sidebar.header('Input Features')
def user_input_features():
    tenure = st.sidebar.slider('Tenure (months)', 1, 72, 12)
    monthly_charges = st.sidebar.slider('Monthly Charges', 20.0, 120.0, 70.0)
    contract = st.sidebar.selectbox('Contract Type', ['Month-to-month', 'One year', 'Two year'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    senior_citizen = st.sidebar.selectbox('Senior Citizen', [0, 1])
    partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
    dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
    internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
    payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    
    data = {
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'contract': contract,
        'gender': gender,
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'internet_service': internet_service,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method
    }

    # Convert to DataFrame
    features = pd.DataFrame(data, index=[0])
    
    # Encode categorical features
    categorical_columns = [
        'contract', 'gender', 'partner', 'dependents',
        'internet_service', 'paperless_billing', 'payment_method'
    ]
    for col in categorical_columns:
        le = encoders[col]
        features[col] = le.transform(features[col])
    
    return features

df = user_input_features()

# Standardize the features
df_scaled = scaler.transform(df)

# Make prediction
prediction = model.predict(df_scaled)

# Display the results
st.subheader('Prediction')
st.write('The predicted churn status is: ', 'Churn' if prediction[0] == 1 else 'No Churn')

# Optionally display the input features
st.subheader('Input Features')
st.write(df)
