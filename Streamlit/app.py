import streamlit as st
import pandas as pd
import joblib

#the model is named 'trained_pipeline.pkl' and located in the same directory)
model_path='hrczggyrgy/ml-obesity-risk-prediction/Streamlit/trained_pipeline.pkl'

@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load(model_path)
    return model

try:
    model = load_model()
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error(f"Failed to load the model. Make sure the model is located at: {model_path}")


st.title('Predict Your Obesity Risk Level')

# Display instructions for the user to follow when providing their details
st.write("Please enter your details to predict your obesity risk level.")

# Create a sidebar for input to make the main screen less cluttered
with st.sidebar:
    st.header("Enter Your Details")

    # For categorical inputs like Gender, Family History, etc., use selectbox for better UX
    Gender = st.selectbox('Gender', ['Male', 'Female'])
    
    Age = st.number_input('Age (years)', min_value=0, max_value=120, step=1, help="Your age in years.")
    
    Height = st.number_input('Height (meters)', min_value=0.0, max_value=3.0, step=0.01, format="%.2f", help="Your height in meters.")
    
    Weight = st.number_input('Weight (kilograms)', min_value=0.0, max_value=500.0, step=0.1, format="%.1f", help="Your weight in kilograms.")
    
    family_history_with_overweight = st.selectbox('Family History of Overweight', ['Yes', 'No'], index=0)
    
    FAVC = st.selectbox('Frequent Consumption of High Caloric Food', ['Yes', 'No'], index=0)
    
    FCVC = st.slider('Frequency of Vegetables Consumption (per day)', min_value=0.0, max_value=3.0, step=0.1, value=1.0, help="0: Never, 3: Very often")
    
    NCP = st.slider('Number of Main Meals', min_value=1, max_value=5, step=1, value=3, help="1 to 5 where 1 is once per day and 5 is five times per day")
    
    CAEC = st.selectbox('Consumption of Food Between Meals', ['No', 'Sometimes', 'Frequently', 'Always'], index=1)
    
    SMOKE = st.selectbox('Do You Smoke?', ['Yes', 'No'], index=1)
    
    CH2O = st.slider('Consumption of Water Daily (liters)', min_value=0.0, max_value=5.0, step=0.1, value=2.0, help="Daily water intake in liters.")
    
    SCC = st.selectbox('Do You Monitor Your Calorie Consumption?', ['Yes', 'No'], index=1)
    
    FAF = st.slider('Physical Activity Frequency (per week)', min_value=0.0, max_value=7.0, step=0.1, value=1.0, help="0: No physical activity, 7: Everyday exercise")
    
    TUE = st.slider('Time Using Technology Devices (hours per day)', min_value=0.0, max_value=24.0, step=0.1, value=2.0)
    
    CALC = st.selectbox('Alcohol Consumption', ['No', 'Sometimes', 'Frequently', 'Always'], index=0)
    
    MTRANS = st.selectbox('Transportation Used', ['Public Transportation', 'Automobile', 'Walking', 'Bike', 'Motorbike'], index=0)

# Button to make predictions
if st.sidebar.button("Submit"):
    # Prepare data for prediction in the same format as the training data
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

    # Note: Ensure the input data is preprocessed (if necessary) in the same way as the training data before prediction
    prediction = model.predict(input_data)
    st.success(f'Predicted Obesity Level: {prediction[0]}')
