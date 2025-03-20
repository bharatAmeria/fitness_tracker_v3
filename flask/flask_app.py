import os
import sys
import pandas as pd
from flask import Flask, request, jsonify
from src.logger import logging
from src.exception import MyException
from load_model import load_model  

# Initialize Flask App
app = Flask(__name__)

# Load the best model from MLflow using model_loader.py
try:
    logging.info("Loading best model using model_loader...")
    model = load_model()
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error("ERROR: Failed to load model.", exc_info=True)
    raise MyException(e, sys)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to predict if the input represents an exercise and classify the type of exercise.
    
    Expected JSON Input:
    {
        "data": [
            {"acc_x": 0.1, "acc_y": 0.2, "acc_z": 0.3, "gyr_x": 0.4, "gyr_y": 0.5, "gyr_z": 0.6}
        ]
    }
    """
    try:
        input_json = request.get_json()
        logging.info(f"Received request: {input_json}")

        if "data" not in input_json:
            return jsonify({"error": "Missing 'data' key in request."}), 400

        # Convert input JSON to DataFrame
        input_data = pd.DataFrame(input_json["data"])
        logging.info(f"Input DataFrame: {input_data.head()}")

        # Make prediction
        predictions = model.predict(input_data)
        logging.info(f"Predictions: {predictions}")

        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        logging.error("ERROR: Failed to make prediction.", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
