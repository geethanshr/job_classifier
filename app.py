import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the scaler and the model
scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

# Define the prediction function
def predict(input_data):
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    return prediction

# Streamlit app
st.title("ML Prediction App")

# Input form
st.header("Enter input data for prediction:")

ssc_p = st.number_input("SSC Percentage")
hsc_p = st.number_input("HSC Percentage")
degree_p = st.number_input("Degree Percentage")
mba_p = st.number_input("MBA Percentage")
etest_p = st.number_input("Etest Percentage")

# Binary columns
gender_M = st.selectbox("Gender (Male=1, Female=0)", [0, 1])
hsc_s_Commerce = st.selectbox("HSC Stream Commerce (Yes=1, No=0)", [0, 1])
hsc_s_Science = st.selectbox("HSC Stream Science (Yes=1, No=0)", [0, 1])
degree_t_Others = st.selectbox("Degree Type Others (Yes=1, No=0)", [0, 1])
degree_t_Sci_Tech = st.selectbox("Degree Type Sci&Tech (Yes=1, No=0)", [0, 1])
workex_Yes = st.selectbox("Work Experience (Yes=1, No=0)", [0, 1])
specialisation_Mkt_HR = st.selectbox("Specialisation Mkt&HR (Yes=1, No=0)", [0, 1])

input_data = pd.DataFrame([[ssc_p, hsc_p, degree_p, mba_p, etest_p, gender_M, hsc_s_Commerce, hsc_s_Science,
                           degree_t_Others, degree_t_Sci_Tech, workex_Yes, specialisation_Mkt_HR]],
                         columns=["ssc_p", "hsc_p", "degree_p", "mba_p", "etest_p", "gender_M", "hsc_s_Commerce",
                                  "hsc_s_Science", "degree_t_Others", "degree_t_Sci&Tech", "workex_Yes", "specialisation_Mkt&HR"])

# Predict button
if st.button("Predict"):
    prediction = predict(input_data)
    st.write("Prediction:", prediction)
