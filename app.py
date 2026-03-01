import os
import pickle
import time
import subprocess
import sys
import json
import re
import threading
from datetime import datetime

import pandas as pd
import requests
import torch
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from transformers import pipeline
from PIL import Image


# =========================================================
# Flask App Initialization
# =========================================================

app = Flask(__name__)

# =========================================================
# Global Configurations
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


# =========================================================
# Lazy Load Leaf Disease Detection Model (IMPORTANT FIX)
# =========================================================

pipe = None

def get_disease_model():
    global pipe

    if pipe is None:
        print("Loading plant disease detection model...")

        pipe = pipeline(
            "image-classification",
            model="nateraw/vit-base-beans",
            device=-1
        )

        print("Model loaded successfully.")

    return pipe


# =========================================================
# Load Agricultural Models
# =========================================================

def get_absolute_path(relative_path):
    return os.path.join(BASE_DIR, relative_path)


try:
    with open(get_absolute_path('agricultural_models.pkl'), 'rb') as f:
        models = pickle.load(f)

    seed_size_model = models['seed_size_model']
    sowing_depth_model = models['sowing_depth_model']
    spacing_model = models['spacing_model']
    label_encoders = models['label_encoders']

    print("Agricultural ML models loaded successfully.")

except Exception as e:

    print("WARNING: agricultural_models.pkl not found or failed")
    print(str(e))

    seed_size_model = None
    sowing_depth_model = None
    spacing_model = None
    label_encoders = {}


# =========================================================
# Load Dropdown Values
# =========================================================

try:
    with open(get_absolute_path('unique_values.pkl'), 'rb') as f:
        unique_values = pickle.load(f)

except Exception as e:

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
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clear_screen():
    pass  # Disabled for cloud


def print_banner():
    print("""
===========================================
 Smart Agriculture Analysis Suite
===========================================
""")


def check_requirements():
    pass  # Disabled for cloud


# =========================================================
# Routes
# =========================================================

@app.route('/')
def home():
    return render_template('homepage.html')


# =========================================================
# Leaf Disease Routes
# =========================================================

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

            model = get_disease_model()

            predictions = model(image)

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

        crops=unique_values.get('Crop Name', []),
        regions=unique_values.get('Region', []),
        seasons=unique_values.get('Season', []),
        soil_types=unique_values.get('Soil Type', [])

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
# Weather API Proxy
# =========================================================

@app.route('/api/weather-proxy')
def weather_proxy():

    try:

        lat = request.args.get('lat')
        lon = request.args.get('lon')

        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"

        response = requests.get(url, timeout=10)

        return jsonify(response.json())

    except Exception as e:

        return jsonify({"error": str(e)}), 500


# =========================================================
# Main Entry (Render / Railway Compatible)
# =========================================================

if __name__ == '__main__':

    print_banner()

    port = int(os.environ.get("PORT", 8000))

    print(f"Starting server on port {port}")

    app.run(

        host="0.0.0.0",
        port=port,
        debug=False

    )