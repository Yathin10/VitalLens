from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import random
import csv
import os
import joblib
import numpy as np
import pandas as pd  # For correct DataFrame handling

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

# --- Load ML Models ---
try:
    bp_model = joblib.load(os.path.join(MODELS_DIR, 'bp_model.pkl'))
    hr_model = joblib.load(os.path.join(MODELS_DIR, 'hr_model.pkl'))
    stress_model = joblib.load(os.path.join(MODELS_DIR, 'stress_model.pkl'))
    print("--- All ML models loaded successfully ---")
except FileNotFoundError:
    print("--- WARNING: Model files not found. /process endpoint will not work ---")
    bp_model = hr_model = stress_model = None

# --- Home Route ---
@app.route('/')
def home():
    return jsonify({'message': 'Welcome to the VitalLens API!'}), 200

# --- Favicon handler ---
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(SCRIPT_DIR, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# --- User Authentication ---
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

# --- ML Prediction Endpoint ---
@app.route('/process', methods=['POST'])
def process_data():
    if not all([bp_model, hr_model, stress_model]):
        return jsonify({'status': 'error', 'message': 'Models are not loaded on the server.'}), 500

    simulated_calculated_hr = random.uniform(60.0, 100.0)
    input_df = pd.DataFrame([[simulated_calculated_hr]], columns=['calculated_hr'])

    bp_pred = bp_model.predict(input_df)
    if bp_pred.ndim == 1:
        bp_pred = bp_pred.reshape(1, -1)

    hr_pred = hr_model.predict(input_df)
    if hr_pred.ndim == 1:
        hr_pred = hr_pred.reshape(1, -1)

    stress_pred = stress_model.predict(input_df)

    results = {
        "systolic": int(bp_pred[0][0]),
        "diastolic": int(bp_pred[0][1]),
        "heartRate": int(hr_pred[0][0]),
        "stress": stress_pred[0]
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
