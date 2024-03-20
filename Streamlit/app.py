import streamlit as st
import pandas as pd
import joblib

# Define the path to the model file
MODEL_PATH = 'trained_pipeline.pkl'

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_model(path):
    model = joblib.load(path)
    return model

model = load_model(MODEL_PATH)

# Streamlit application title
st.title('Obesity Level Prediction')

# User inputs via sidebar
with st.sidebar:
    st.header("Please enter your details:")
    
    # Input fields matched to the model's expectations
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.number_input('Age', step=1, format="%d")
    height = st.number_input('Height (in meters)', format="%.2f")
    weight = st.number_input('Weight (in kilograms)', format="%.2f")
    family_history = st.selectbox('Family history of overweight', ['Yes', 'No'])
    favc = st.selectbox('Frequent consumption of high caloric food', ['Yes', 'No'])
    fcvc = st.number_input('Frequency of consumption of vegetables', format="%.1f")
    ncp = st.number_input('Number of main meals', format="%.1f")
    caec = st.selectbox('Consumption of food between meals', ['No', 'Sometimes', 'Frequently', 'Always'])
    smoke = st.selectbox('Do you smoke?', ['Yes', 'No'])
    ch2o = st.number_input('Consumption of water daily', format="%.1f")
    scc = st.selectbox('Calories consumption monitoring', ['Yes', 'No'])
    faf = st.number_input('Physical activity frequency', format="%.1f")
    tue = st.number_input('Time using technology devices', format="%.1f")
    calc = st.selectbox('Alcohol consumption', ['No', 'Sometimes', 'Frequently', 'Always'])
    mtrans = st.selectbox('Main mode of transportation', ['Public Transportation', 'Bike', 'Walking', 'Automobile', 'Motorbike'])

# Predict button
if st.sidebar.button('Predict Obesity Level'):
    # Create a DataFrame for the model input
    input_data = pd.DataFrame([[gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]],
                              columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'])

    # Prediction
    prediction = model.predict(input_data)
    
    # Display the prediction result
    st.success(f'Predicted Obesity Level: {prediction[0]}')

# Note to the user about the prediction
st.info('This prediction is based on a machine learning model and should not be used as medical advice.')
