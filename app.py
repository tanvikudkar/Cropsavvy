import os
import subprocess
import sys
import time
import threading
import webbrowser
import pickle
import pandas as pd
import json
import re
from flask import Flask, request, render_template, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from transformers import pipeline
from PIL import Image
import torch
from datetime import datetime
import requests

app = Flask(__name__)

# --- Global Configurations ---
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# =========================================================
# FIXED Leaf Disease Detection Model Initialization
# =========================================================

print("Loading plant disease detection model...")

pipe = pipeline(
    "image-classification",
    model="nateraw/vit-base-beans",
    device=-1  # CPU
)

print("Model loaded successfully.")

# =========================================================
# Seed Size Model Initialization
# =========================================================

current_app_dir = os.path.dirname(os.path.abspath(__file__))

def get_absolute_path(relative_path):
    return os.path.join(current_app_dir, relative_path)

try:
    with open(get_absolute_path('agricultural_models.pkl'), 'rb') as f:
        models = pickle.load(f)

    seed_size_model = models['seed_size_model']
    sowing_depth_model = models['sowing_depth_model']
    spacing_model = models['spacing_model']
    label_encoders = models['label_encoders']

    print("Agricultural ML models loaded successfully.")

except FileNotFoundError:

    print("WARNING: agricultural_models.pkl not found")

    seed_size_model = None
    sowing_depth_model = None
    spacing_model = None
    label_encoders = {}

# Load dropdown values
try:
    with open(get_absolute_path('unique_values.pkl'), 'rb') as f:
        unique_values = pickle.load(f)

except FileNotFoundError:

    print("WARNING: unique_values.pkl not found")

    unique_values = {
        'Crop Name': [],
        'Region': [],
        'Season': [],
        'Soil Type': []
    }

# =========================================================
# Helper Functions
# =========================================================

def allowed_file(filename):

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_screen():

    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


def print_banner():

    print("""
===============================================
 Agricultural Analysis Suite - Launch Tool
===============================================
""")


def check_requirements():

    try:
        import flask
        print("Flask OK")
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        )


def wait_for_exit():

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")


# =========================================================
# Routes
# =========================================================

@app.route('/')
def home():

    return render_template('homepage.html')


@app.route('/leaf_disease')
def leaf_index():

    return render_template('leaf_disease_index.html')


@app.route('/leaf_disease/predict', methods=['POST'])
def leaf_predict():

    if 'file' not in request.files:

        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':

        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):

        filename = secure_filename(file.filename)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        file.save(filepath)

        try:

            image = Image.open(filepath).convert("RGB")

            predictions = pipe(image)

            formatted_predictions = []

            for pred in predictions:

                formatted_predictions.append({

                    "disease": pred["label"],

                    "confidence": f"{pred['score']*100:.2f}%",

                    "score": float(pred["score"])

                })

            return jsonify({

                "success": True,

                "predictions": formatted_predictions,

                "top_prediction": formatted_predictions[0],

                "image_path": f"/static/uploads/{filename}"

            })

        except Exception as e:

            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "Invalid file"}), 400


# =========================================================
# Seed Size Routes
# =========================================================

@app.route('/seed_size')
def seed_index():

    return render_template(

        'seed_size_index.html',

        crops=unique_values['Crop Name'],

        regions=unique_values['Region'],

        seasons=unique_values['Season'],

        soil_types=unique_values['Soil Type']
    )


@app.route('/seed_size/predict', methods=['POST'])
def seed_predict():

    if seed_size_model is None:

        return jsonify({"error": "ML model not loaded"}), 500

    try:

        crop_name = request.form['crop_name']

        region = request.form['region']

        season = request.form['season']

        temperature = float(request.form.get('temperature', 0))

        moisture = float(request.form.get('moisture', 0))

        soil_type = request.form['soil_type']

        soil_ph = float(request.form['soil_ph'])

        crop_encoded = label_encoders['Crop Name'].transform([crop_name])[0]

        region_encoded = label_encoders['Region'].transform([region])[0]

        season_encoded = label_encoders['Season'].transform([season])[0]

        soil_encoded = label_encoders['Soil Type'].transform([soil_type])[0]

        features = [[

            crop_encoded,
            region_encoded,
            season_encoded,
            temperature,
            moisture,
            soil_encoded,
            soil_ph
        ]]

        seed_size_encoded = seed_size_model.predict(features)[0]

        sowing_depth = sowing_depth_model.predict(features)[0]

        spacing = spacing_model.predict(features)[0]

        seed_size = label_encoders['Seed Size Category'].inverse_transform(

            [seed_size_encoded]

        )[0]

        return jsonify({

            "seed_size": seed_size,

            "sowing_depth": round(float(sowing_depth), 2),

            "spacing": round(float(spacing), 2)

        })

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# =========================================================
# Weather API
# =========================================================

@app.route('/api/weather-proxy')
def weather_proxy():

    try:

        lat = request.args.get('lat')

        lon = request.args.get('lon')

        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

        response = requests.get(url, timeout=5)

        data = response.json()

        return jsonify(data)

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# =========================================================
# Main
# =========================================================

if __name__ == '__main__':

    clear_screen()

    print_banner()

    check_requirements()

    threading.Thread(

        target=lambda: (

            time.sleep(1),

            webbrowser.open("http://localhost:8000")

        )

    ).start()

    app.run(

        host="0.0.0.0",

        port=8000,

        debug=False
    )

    wait_for_exit()