import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Function to load pickled objects (model and scaler)
def load_pickled_objects(model_file, scaler_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Load your model and scaler
model_file = 'model.pkl'  # Replace with your actual file path
scaler_file = 'scaler.pkl'  # Replace with your actual file path
model, scaler = load_pickled_objects(model_file, scaler_file)

# Streamlit app
def main():
    st.title('Your Classification Model Deployment')

    # Collect user input
    ssc_p = st.slider('Secondary Education Percentage', 0.0, 100.0, 50.0)
    hsc_p = st.slider('Higher Secondary Education Percentage', 0.0, 100.0, 50.0)
    degree_p = st.slider('Degree Percentage', 0.0, 100.0, 50.0)
    mba_p = st.slider('MBA Percentage', 0.0, 100.0, 50.0)
    etest_p = st.slider('Employability Test Percentage', 0.0, 100.0, 50.0)

    gender_M = st.checkbox('Male Gender')
    hsc_s_Commerce = st.checkbox('Higher Secondary in Commerce')
    hsc_s_Science = st.checkbox('Higher Secondary in Science')
    degree_t_Others = st.checkbox('Degree in Others')
    degree_t_SciTech = st.checkbox('Degree in Science & Tech')
    workex_Yes = st.checkbox('Work Experience')
    specialisation_MktHR = st.checkbox('Specialisation in Marketing & HR')

    # Prepare user input for prediction
    user_input = pd.DataFrame({
        'ssc_p': [ssc_p],
        'hsc_p': [hsc_p],
        'degree_p': [degree_p],
        'mba_p': [mba_p],
        'etest_p': [etest_p],
        'gender_M': [gender_M],
        'hsc_s_Commerce': [hsc_s_Commerce],
        'hsc_s_Science': [hsc_s_Science],
        'degree_t_Others': [degree_t_Others],
        'degree_t_Sci&Tech': [degree_t_SciTech],
        'workex_Yes': [workex_Yes],
        'specialisation_Mkt&HR': [specialisation_MktHR]
    })

    # Scale the specific input columns
    columns_to_scale = ['ssc_p', 'hsc_p', 'degree_p', 'mba_p', 'etest_p']
    user_input_scaled = user_input.copy()
    user_input_scaled[columns_to_scale] = scaler.transform(user_input[columns_to_scale])

    # Make predictions
    prediction = model.predict(user_input_scaled)

    # Display prediction
    st.write(f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    main()
