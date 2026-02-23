
# app.py
from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash
import requests
import numpy as np
import torch
import csv
import os
from transformers import BertModel, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import pandas as pd
from new_model import predict_yield
from login_manager import UserAuthenticator

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_hackaccino'  # Change this in production

# Initialize Authenticator
authenticator = UserAuthenticator(os.path.join(os.path.dirname(__file__), 'user_credentials.csv'))

# --- Configuration & Setup ---

# Load pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
use_fake_embedding = False
try:
    tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
    bert_model = BertModel.from_pretrained(model_name, local_files_only=True)
except Exception:
    tokenizer = None
    bert_model = None
    use_fake_embedding = True

# --- Helper Functions ---

# Crop requirements data: NPK (N-P-K), pH, and description
CROP_REQUIREMENTS = {
    "Rice": {"npk": "80-40-40", "ph": "5.5-6.5", "desc": "High moisture requirement, thrives in alluvial soil."},
    "Wheat": {"npk": "120-60-40", "ph": "6.0-7.0", "desc": "Cool climate, well-drained loamy soil."},
    "Cotton": {"npk": "100-50-50", "ph": "6.0-7.5", "desc": "Deep black soil or alluvial soil, warm climate."},
    "Sugarcane": {"npk": "150-80-80", "ph": "6.5-7.5", "desc": "Deep rich loamy soil, abundant water."},
    "Tea": {"npk": "120-60-60", "ph": "4.5-5.5", "desc": "Well-drained acidic soil, hilly terrain."},
    "Fruits": {"npk": "60-30-30", "ph": "6.0-7.0", "desc": "Varies by type, generally well-drained soil."},
    "Vegetables": {"npk": "80-40-60", "ph": "6.0-7.0", "desc": "Rich organic matter, consistent moisture."},
    "Soybean": {"npk": "20-60-40", "ph": "6.0-7.0", "desc": "Warm climate, well-drained loamy soil."},
    "Apples": {"npk": "50-30-50", "ph": "6.0-6.5", "desc": "Cool temperate climate, fertile loamy soil."},
    "Rubber": {"npk": "40-40-10", "ph": "4.5-5.5", "desc": "Tropical climate, well-drained acidic soil."},
    "Cashew": {"npk": "40-20-20", "ph": "5.5-6.5", "desc": "Coastal regions, well-drained sandy or lateritic soil."},
    "Coconut": {"npk": "80-60-120", "ph": "5.5-7.0", "desc": "Coastal sandy soil, high humidity."}
}

def get_detailed_recommendation(state=None, soil_type=None):
    file_path = os.path.join(os.path.dirname(__file__), 'Data.csv')
    options = []
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                match = False
                if state and row["State"].lower() == state.lower():
                    match = True
                elif soil_type and row["Soil"].lower() == soil_type.lower():
                    match = True
                
                if match:
                    crop = row["Crop"]
                    req = CROP_REQUIREMENTS.get(crop, {"npk": "80-40-40", "ph": "6.0-7.0", "desc": "Standard requirements."})
                    
                    # Calculate average potential yield from yield_data.csv
                    potential_yield = "N/A"
                    try:
                        yd_path = os.path.join(os.path.dirname(__file__), 'yield_data.csv')
                        if os.path.exists(yd_path):
                            yd_df = pd.read_csv(yd_path)
                            avg_yield = yd_df[yd_df['Crop'] == crop]['Yield_Tonnes_Per_Acre'].mean()
                            if not np.isnan(avg_yield):
                                potential_yield = f"{round(avg_yield, 2)} Tonnes/Acre"
                    except Exception:
                        pass

                    options.append({
                        "state": row["State"],
                        "soil": row["Soil"],
                        "crop": crop,
                        "npk": req["npk"],
                        "ph": req["ph"],
                        "desc": req["desc"],
                        "potential_yield": potential_yield
                    })
    except FileNotFoundError:
        return None
    return options

def get_soil_treatment_advice(crop, land_area, current_ph):
    """Calculates fertilizer and pH correction advice based on pH.py logic."""
    target_ph_map = {
        "Rice": 5.6, "Sugarcane": 5.5, "Cotton": 6.25, "Fruits": 3.5, 
        "Tea": 7.0, "Vegetables": 6.5, "Cashew": 6.5, "Apples": 6.5, 
        "Rubber": 5.25, "Soybean": 6.5, "Wheat": 6.5
    }
    
    target_ph = target_ph_map.get(crop, 6.5) 
    advice = {
        "status": "Optimal",
        "warning": "",
        "recommendation": "You have the perfect pH concentration for this crop.",
        "materials": []
    }

    if current_ph > target_ph:
        if current_ph - target_ph >= 1:
            advice["status"] = "High Alkalinity"
            advice["warning"] = "You may lose 10%-50% of your yield due to high pH."
            advice["recommendation"] = "Apply acidifying materials to lower the pH."
            advice["materials"] = [
                {"name": "Ammonium sulfate", "amount": round(land_area * 75, 1), "unit": "lbs"},
                {"name": "Elemental sulfur", "amount": round(land_area * 150, 1), "unit": "lbs"},
                {"name": "Sulfur-coated urea", "amount": round(land_area * 75, 1), "unit": "lbs"},
                {"name": "Aluminum sulfate", "amount": round(land_area * 15, 1), "unit": "lbs"}
            ]
        else:
            advice["status"] = "Slightly Alkaline"
            advice["warning"] = "You may lose more than 1% of your yield."
            advice["recommendation"] = "Incorporate organic matter to naturally balance the soil."
    elif target_ph > current_ph:
        if target_ph - current_ph >= 1:
            advice["status"] = "High Acidity"
            advice["warning"] = "You may lose 10%-50% of your yield due to low pH."
            advice["recommendation"] = "Apply liming materials to raise the pH."
            advice["materials"] = [
                {"name": "Agricultural lime", "amount": round(land_area * 3000, 1), "unit": "lbs"},
                {"name": "Dolomitic lime", "amount": round(land_area * 3000, 1), "unit": "lbs"},
                {"name": "Wood ash", "amount": round(land_area * 750, 1), "unit": "lbs"},
                {"name": "Gypsum", "amount": round(land_area * 750, 1), "unit": "lbs"},
                {"name": "Calcium carbonate", "amount": round(land_area * 750, 1), "unit": "lbs"}
            ]
        else:
            advice["status"] = "Slightly Acidic"
            advice["warning"] = "You may lose more than 1% of your yield."
            advice["recommendation"] = "Incorporate organic matter to naturally balance the soil."
            
    return advice

# --- Routes ---

@app.route('/')
def login():
    if 'username' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form.get('email') # Using email as username for now based on template
    password = request.form.get('password')
    
    if authenticator.authenticate_user(username, password):
        session['username'] = username
        return redirect(url_for('home'))
    else:
        flash('Invalid email or password', 'error')
        return redirect(url_for('login'))

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('email') # Using email as username
    password = request.form.get('password')
    name = request.form.get('name') # Captured but not stored in simple CSV auth for now, or could modify CSV
    
    # Simple check if user exists could be added here, but for now just register
    authenticator.register_user(username, password)
    
    # Auto login after register
    session['username'] = username
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/yield')
def yield_page():
    # Handle pre-filled crop and location
    crop = request.args.get('crop', '')
    location = request.args.get('location', '')
    return render_template('yield.html', prefilled_crop=crop, prefilled_location=location)

@app.route('/recommend')
def recommend_page():
    return render_template('recommend.html')

@app.route('/predict_yield', methods=['POST'])
def predict_yield_route():
    # Capture form inputs
    location = request.form.get('location', 'Unknown')
    land_area = float(request.form.get('landArea', 1.0))
    soil_ph = float(request.form.get('soilPH', 6.5))
    crop_type = request.form.get('cropType', 'Rice')
    season = request.form.get('season', 'Whole Year')
    n_val = float(request.form.get('nitrogen', 80))
    p_val = float(request.form.get('phosphorus', 40))
    k_val = float(request.form.get('potassium', 40))
    
    # Weatherstack API parameters
    api_key = "6f9c6496403400f57c4fded4f46cad0c" 

    weather_features = [25, 0, 50, 10, 5] # Default fallback
    weather_info = None

    try:
        weather_api_url = f"http://api.weatherstack.com/current?access_key={api_key}&query={location}"
        response = requests.get(weather_api_url)
        weather_data = response.json()
        
        if 'current' in weather_data:
            weather_features = [
                weather_data["current"]["temperature"],
                weather_data["current"]["precip"],
                weather_data["current"]["humidity"],
                weather_data["current"]["cloudcover"],
                weather_data["current"]["wind_speed"],
            ]
            weather_info = {
                "temp": weather_data["current"]["temperature"],
                "desc": weather_data["current"]["weather_descriptions"][0],
                "humidity": weather_data["current"]["humidity"],
                "wind": weather_data["current"]["wind_speed"],
                "location": f"{weather_data['location']['name']}, {weather_data['location']['country']}"
            }
    except Exception:
         pass

    # Predict using centralized model logic
    try:
        final_yield_per_acre = predict_yield(
            crop=crop_type,
            season=season,
            n=n_val,
            p=p_val,
            k=k_val,
            ph=soil_ph,
            temp=weather_features[0],
            humidity=weather_features[2],
            rainfall=weather_features[1] * 10
        )
        
        if final_yield_per_acre is None:
            raise Exception("Prediction returned None")
            
    except Exception as e:
        # Fallback to simulated logic if model loading fails
        base_prediction = 2.0
        final_yield_per_acre = base_prediction * (n_val/80) * (soil_ph/6.5)

    total_yield = final_yield_per_acre * land_area
    
    # Category Assignment (based on common yield ranges)
    category = "Moderate"
    if final_yield_per_acre > 5.0: category = "Exceptional"
    elif final_yield_per_acre > 3.0: category = "High"
    elif final_yield_per_acre < 1.0: category = "Low"

    # Load Metrics for UI
    metrics = {"mse": "N/A", "r2": "N/A"}
    try:
        metrics_path = os.path.join(os.path.dirname(__file__), 'model_metrics.txt')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if "Mean Squared Error" in line:
                         metrics["mse"] = line.split(": ")[1].strip()
                    if "R2 Score" in line:
                         metrics["r2"] = line.split(": ")[1].strip()
    except Exception:
        pass

    # Get Soil Treatment Advice
    treatment_advice = get_soil_treatment_advice(crop_type, land_area, soil_ph)

    return render_template('predict_yield_result.html', 
                           predicted_yield=round(total_yield, 2),
                           yield_per_acre=round(final_yield_per_acre, 2),
                           category=category,
                           location=location,
                           land_area=land_area,
                           soil_ph=soil_ph,
                           crop_type=crop_type,
                           season=season,
                           npk=f"{n_val}-{p_val}-{k_val}",
                           weather=weather_info,
                           metrics=metrics,
                           treatment=treatment_advice)

@app.route('/get_recommendation', methods=['POST'])
def get_recommendation_route():
    state = request.form.get('state')
    soil = request.form.get('soil')
    
    if not state and not soil:
        return render_template('recommend.html', error="Please enter a state or soil type.")
    
    results = get_detailed_recommendation(state=state, soil_type=soil)
    
    if not results:
        return render_template('recommend.html', error="No recommendations found for the given input.", state=state, soil=soil)
    
    return render_template('recommend.html', results=results, state=state, soil=soil)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
