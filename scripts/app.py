
import streamlit as st
import pandas as pd
import pickle

# Load your trained model
with open('trained_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

st.title('Prediction Application')

# Create form for user input
with st.form("prediction_form"):
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    Age = st.number_input('Age', format="%f", min_value=0.0, max_value=100.0)
    Height = st.number_input('Height (in meters)', format="%f", min_value=0.0, help="Please enter your height in meters (m)")
    Weight = st.number_input('Weight (in kilograms)', format="%f", min_value=0.0, help="Please enter your weight in kilograms (kg)")
    family_history_with_overweight = st.selectbox('Family History with Overweight', ['yes', 'no'])
    FAVC = st.selectbox('Frequent consumption of high caloric food', ['yes', 'no'])
    FCVC = st.number_input('Frequency of vegetables consumption', format="%f", min_value=0.0)
    NCP = st.number_input('Number of main meals', format="%f", min_value=0.0)
    CAEC = st.selectbox('Consumption of food between meals', ['no', 'Sometimes', 'Frequently', 'Always'])
    SMOKE = st.selectbox('Smoke', ['no', 'yes'])
    CH2O = st.number_input('Consumption of water daily', format="%f", min_value=0.0)
    SCC = st.selectbox('Calories consumption monitoring', ['no', 'yes'])
    FAF = st.number_input('Physical activity frequency', format="%f", min_value=0.0)
    TUE = st.number_input('Time using technology devices', format="%f", min_value=0.0)
    CALC = st.selectbox('Alcohol consumption', ['no', 'Sometimes', 'Frequently', 'Always'])
    MTRANS = st.selectbox('Transportation used', ['Public_Transportation', 'Automobile', 'Walking', 'Bike', 'Motorbike'])
    
    submitted = st.form_submit_button("Submit")
    if submitted:
        # Prepare data for prediction
        input_data = pd.DataFrame({
            'Gender': [Gender],
            'Age': [Age],
            'Height': [Height],
            'Weight': [Weight],
            'family_history_with_overweight': [family_history_with_overweight],
            'FAVC': [FAVC],
            'FCVC': [FCVC],
            'NCP': [NCP],
            'CAEC': [CAEC],
            'SMOKE': [SMOKE],
            'CH2O': [CH2O],
            'SCC': [SCC],
            'FAF': [FAF],
            'TUE': [TUE],
            'CALC': [CALC],
            'MTRANS': [MTRANS],
        })

        # Get model prediction
        prediction = model.predict(input_data)

        st.write(f'Prediction: {prediction[0]}')
    