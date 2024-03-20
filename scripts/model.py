# Flask Application code with prediction model
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load your trained model
with open('trained_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Collect form data as Python dictionary
        input_data = {
            'Gender': [request.form['Gender']],
            'Age': [float(request.form['Age'])],
            'Height': [float(request.form['Height'])],
            'Weight': [float(request.form['Weight'])],
            'family_history_with_overweight': [request.form['family_history_with_overweight']],
            'FAVC': [request.form['FAVC']],
            'FCVC': [float(request.form['FCVC'])],
            'NCP': [float(request.form['NCP'])],
            'CAEC': [request.form['CAEC']],
            'SMOKE': [request.form['SMOKE']],
            'CH2O': [float(request.form['CH2O'])],
            'SCC': [request.form['SCC']],
            'FAF': [float(request.form['FAF'])],
            'TUE': [float(request.form['TUE'])],
            'CALC': [request.form['CALC']],
            'MTRANS': [request.form['MTRANS']],
        }

        # Convert to pandas DataFrame
        df = pd.DataFrame.from_dict(input_data)

        # Get model prediction
        prediction = model.predict(df)

        return render_template('result.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)