import tensorflow as tf
import pickle
import numpy as np
import os
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pathlib import Path
from typing import Literal
from tensorflow.keras.models import load_model

# Define fastapi app
app = FastAPI()

# Define models
MODEL_DIR = Path(__file__).parent.parent.parent  / "models"
MODEL_NAME = Literal['Linear Regression', 'Decision Tree', 'XGBoost', 'Neural Network']

# 1. Define models definitely
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
model_nn_price = load_model(MODEL_DIR / "model_price_nn.h5")
model_nn_time = load_model(MODEL_DIR / "model_time_nn.h5")
feature_list = joblib.load(MODEL_DIR / "features.pkl")
le_pickup = joblib.load(MODEL_DIR / "le_pickup.pkl")
le_drop = joblib.load(MODEL_DIR / "le_drop.pkl")

# 2. Load specified .pkl models
"""Load models price and trip"""
model_trip_ml = joblib.load(MODEL_DIR / "best_models_trip.pkl")


# 3. Define input SCHEMA with features
class RideRequest(BaseModel):
    pickup: str
    drop: str
    distance: float
    hour: int
    day_of_week: int
    driver_rating: float
    customer_rating: float
    route_avg_ctat: float
    route_avg_price: float
    route_avg_distance: float
    route_count: int
    model: MODEL_NAME

# 4. Features builder function
def build_features(data: RideRequest) -> np.ndarray:
    try:
        pickup_encoded = le_pickup.transform([data.pickup])[0]
        drop_encoded = le_drop.transform([data.drop])[0]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    hour, day = data.hour, data.day_of_week
    distance = data.distance
    vec = [
        pickup_encoded, drop_encoded, hour, day,
        data.driver_rating, data.customer_rating,
        np.log1p(distance), # Log-transform distance
        distance / (hour + 1), # Distance per hour
        int(7 <= hour <= 9 or 17 <= hour <= 19), # is_peak_hour
        int(hour >= 22 or hour < 5), # is_night_time
        np.sin(2 * np.pi * hour/24), # hour_sin
        np.cos(2 * np.pi * hour/24),  # hour_cos
        np.cos(2 * np.pi * day/7),   # day_cos
        data.driver_rating - data.customer_rating, # Data diff
        data.driver_rating * data.customer_rating, # Data interaction
        (data.driver_rating + data.customer_rating) / 2, # Data average
        int(data.driver_rating < 3.5), # low driver rating
        int(data.customer_rating < 3.5), # low customer rating
        data.route_avg_ctat, data.route_avg_price,
        data.route_avg_distance, data.route_count
    ]
    return np.array(vec).reshape(1, -1)

def predict(models_dict, nn_model, ml_model, data: RideRequest) -> float:
    X_raw = build_features(data=data)
    if data.model == 'Neural Network':
        return float(nn_model.predict(X_raw)[0][0])
    
    elif data.model in ['Linear Regression', 'Decision Tree', 'XGBoost']:
        X_scaled = scaler.transform(X_raw)
        return float(ml_model.predict(X_scaled)[0])
    
# 5. Define API endpoint
app.get("/health")
def health_check():
    return {"status": "ok"}, 200

app.get("/features")
def get_features():
    return {"features": feature_list}, 200

