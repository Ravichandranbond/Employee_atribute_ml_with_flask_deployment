from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load model and transformers
model = pickle.load(open('attrition_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define feature groups
categorical_cols = ['jobType', 'degree', 'major', 'industry']
numerical_cols = ['yearsExperience', 'milesFromMetropolis']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form input
        jobType = request.form['jobType']
        degree = request.form['degree']
        major = request.form['major']
        industry = request.form['industry']
        yearsExperience = float(request.form['yearsExperience'])
        milesFromMetropolis = float(request.form['milesFromMetropolis'])

        # Create DataFrame from input
        sample = pd.DataFrame([{
            'jobType': jobType,
            'degree': degree,
            'major': major,
            'industry': industry,
            'yearsExperience': yearsExperience,
            'milesFromMetropolis': milesFromMetropolis
        }])

        # Transform input
        sample_cat = encoder.transform(sample[categorical_cols])
        sample_num = scaler.transform(sample[numerical_cols])
        sample_final = np.concatenate((sample_cat, sample_num), axis=1)

        # Predict salary
        predicted_salary = model.predict(sample_final)[0]
        salary_output = round(predicted_salary, 2)

        return render_template('index.html', prediction_text=f'Predicted Salary: ${salary_output}k')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)
