import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model and preprocessing objects
@st.cache_resource
def load_model_and_objects():
    model = tf.keras.models.load_model('model.h5')
    with open('OHE.pkl', 'rb') as file:
        ohe = pickle.load(file)
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, ohe, label_encoder_gender, scaler

model, ohe, label_encoder_gender, scaler = load_model_and_objects()

# Streamlit app title
st.title("Customer Churn Prediction")

# User input fields
st.header("Enter Customer Details")
geography = st.selectbox(
    "Select Geography",
    options=ohe.categories_[0]  # Options from the trained encoder
)

gender = st.selectbox(
    "Select Gender",
    options=label_encoder_gender.classes_
)

age = st.slider(
    "Select Age",
    min_value=18,
    max_value=92
)

balance = st.number_input(
    "Enter Balance",
    min_value=0.0,
    step=0.01,
    format="%.2f"
)

credit_score = st.number_input(
    "Enter Credit Score",
    min_value=300,  # Assuming a realistic minimum credit score
    max_value=850,
    step=1
)

estimated_salary = st.number_input(
    "Enter Estimated Salary",
    min_value=0.0,
    step=0.01,
    format="%.2f"
)

tenure = st.slider(
    "Select Tenure (Years)",
    min_value=0,
    max_value=10
)

num_of_products = st.slider(
    "Select Number of Products",
    min_value=1,
    max_value=4
)

has_cr_card = st.selectbox(
    "Do you have a Credit Card?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

is_active_member = st.selectbox(
    "Are you an Active Member?",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Prepare the input DataFrame
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# Encoding Geography and Gender
geo_encoded = ohe.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, columns=ohe.get_feature_names_out(["Geography"])
)

input_data["Gender"] = label_encoder_gender.transform(input_data["Gender"])
input_data = input_data.drop("Geography", axis=1)
input_data = pd.concat([input_data, geo_encoded_df], axis=1)

# Normalize numerical features
scaled_features = scaler.transform(input_data)
input_data_scaled = pd.DataFrame(scaled_features, columns=input_data.columns)

# Make predictions
pred = model.predict(input_data_scaled)
pred_prob = pred[0][0]

# Display results
st.subheader("Prediction Result")
if pred_prob > 0.5:
    st.write(f"The customer is likely to churn with a probability of {pred_prob:.2%}.")
else:
    st.write(f"The customer is unlikely to churn with a probability of {(1 - pred_prob):.2%}.")
