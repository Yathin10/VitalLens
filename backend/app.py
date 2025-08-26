from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import csv
import os
import joblib
import numpy as np
import pandas as pd  # Added for proper feature names

# --- App & DB Setup ---
app = Flask(__name__)
CORS(app)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, 'users.csv')
CSV_HEADERS = ['name', 'phone', 'email', 'gender', 'password']
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

def initialize_database():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)
        print(f"Database file '{CSV_FILE}' created.")

initialize_database()

# --- 1. LOAD THE TRAINED MODELS ON STARTUP ---
try:
    bp_model = joblib.load(os.path.join(MODELS_DIR, 'bp_model.pkl'))
    hr_model = joblib.load(os.path.join(MODELS_DIR, 'hr_model.pkl'))
    stress_model = joblib.load(os.path.join(MODELS_DIR, 'stress_model.pkl'))
    print("--- All ML models loaded successfully ---")
    print(f"BP model type: {type(bp_model).__name__}")
    print(f"HR model type: {type(hr_model).__name__}")
    print(f"Stress model type: {type(stress_model).__name__}")
except FileNotFoundError:
    print("--- WARNING: Model files not found. The '/process' endpoint will not work. ---")
    print("--- Please run train_model.py to create the models. ---")
    bp_model = hr_model = stress_model = None

# --- User Authentication Endpoints ---
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    email = data.get('email')
    
    with open(CSV_FILE, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['email'] == email:
                return jsonify({'status': 'error', 'message': 'This email is already registered'}), 409

    with open(CSV_FILE, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        new_user = {
            'name': data.get('name'), 'phone': data.get('phone'),
            'email': email, 'gender': data.get('gender'),
            'password': data.get('password')
        }
        writer.writerow(new_user)
    
    return jsonify({'status': 'success', 'message': 'Account created successfully!'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    with open(CSV_FILE, mode='r', newline='') as f:
        reader = csv.DictReader(f)
        for user in reader:
            if user['email'] == email and user['password'] == password:
                return jsonify({'status': 'success', 'message': 'Login successful'})
    
    return jsonify({'status': 'error', 'message': 'Invalid email or password'}), 401

# --- 2. UPDATED DATA PROCESSING ENDPOINT ---
@app.route('/process', methods=['POST'])
def process_data():
    """
    Endpoint uses the loaded ML models to make predictions.
    Handles different model types and ensures correct input/output shapes.
    """
    if not all([bp_model, hr_model, stress_model]):
        return jsonify({'status': 'error', 'message': 'Models are not loaded on the server.'}), 500

    print("Backend: Received request on /process endpoint.")
    
    # --- SIMULATE LIVE DATA PROCESSING ---
    simulated_calculated_hr = random.uniform(60.0, 100.0)
    print(f"Simulated feature (calculated HR): {simulated_calculated_hr:.2f} bpm")
    
    # --- FIX: Pass DataFrame with correct feature name used in training ---
    input_df = pd.DataFrame([[simulated_calculated_hr]], columns=['calculated_hr'])

    # --- PREDICT WITH LOADED MODELS ---
    # BP Prediction
    bp_pred = bp_model.predict(input_df)
    if bp_pred.ndim == 1:
        bp_pred = bp_pred.reshape(1, -1)

    # HR Prediction
    hr_pred = hr_model.predict(input_df)
    if hr_pred.ndim == 1:
        hr_pred = hr_pred.reshape(1, -1)

    # Stress Prediction
    stress_pred = stress_model.predict(input_df)

    # --- FORMAT RESULTS ---
    results = {
        "systolic": int(bp_pred[0][0]),
        "diastolic": int(bp_pred[0][1]),
        "heartRate": int(hr_pred[0][0]),
        "stress": stress_pred[0]
    }

    print(f"Model Predictions: {results}")
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
