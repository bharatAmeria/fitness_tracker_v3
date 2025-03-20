import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify
from src.logger import logging
from src.exception import MyException

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Load model
MODEL_PATH = "artifact/03_20_2025_18_21_31/model.pkl/random_forest_model.pkl"
try:
    logging.info("Loading trained model...")
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("ERROR: Failed to load model.")
    raise MyException(e, sys)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        data = {key: float(request.form[key]) for key in ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']}
        
        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(df)[0]

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        logging.error("Prediction failed.", exc_info=True)
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)
