import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Function to load custom CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS
load_css('style.css')

# Load your trained RandomForest model
model_path = r"C:\Users\maddi\IODLabs\Capstone\random_forest_model_2.pkl"

try:
    model = load(model_path)
except ImportError as e:
    st.error(f"Error importing necessary libraries: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

st.title('Drone Deployment Predictor')

# Create input fields for features
wind_speed = st.number_input('Wind Speed (km/h)', min_value=0.0, max_value=100.0, value=10.0, step=0.1)
temperature = st.number_input('2m Celsius (°C)', min_value=-50.0, max_value=50.0, value=20.0, step=0.1)
precipitation = st.number_input('TP mm per hour', min_value=0.0, max_value=100.0, value=0.0, step=0.1)

# Button to make predictions
if st.button('Predict Drone Deployment'):
    # Create a DataFrame from the input features
    input_data = pd.DataFrame({
        'wind_speed': [wind_speed],
        '2m_celsius': [temperature],
        'TP_mm_per_hour': [precipitation]
    })

    # Make prediction
    prediction = model.predict(input_data)[0]  # Get the first element of the prediction array

    # Mapping of prediction values to descriptive text
    prediction_map = {
        0: 'Deploy Moderate Capability Drone',
        1: 'Deploy Advanced Capability Drone',
        2: 'Deploy Premium Capability Drone',
        3: 'Do Not Deploy Drone'
    }

    # Display the prediction
    prediction_text = prediction_map.get(prediction, "Unknown Prediction")
    st.write(f'Prediction: {prediction_text}')


