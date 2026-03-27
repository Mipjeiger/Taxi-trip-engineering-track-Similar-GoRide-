import os
import logging
import tensorflow as tf
import joblib
from fastapi import FastAPI
from typing import Literal
from pathlib import Path
from tensorflow.keras.models import load_model


MODEL_DIR_2 = Path(__file__).parent.parent.parent / "models_2"
MODEL_NAME = Literal['Linear Regression', 'Decision Tree', 'XGBoost', 'Neural Network']

best_models = joblib.load(MODEL_DIR_2 / ["best_model_ctat_ultra.pkl", "best_model_vtat_ultra.pkl", "best_models_ultra_ctat.pkl", "best_models_ultra_vtat.pkl"][2])  # FIX 0: was best_models.pkl, now we have two sets of models for price and time
features_list = joblib.load(MODEL_DIR_2 / ["features_new.pkl", "features_ultra.pkl", "features.pkl"][0])
le_list = joblib.load(MODEL_DIR_2 / ["le_drop.pkl", "le_pickup.pkl"][0])
keras_models = {
    "price model": load_model(MODEL_DIR_2 / "model_price_improved.keras"),
    "time model": load_model(MODEL_DIR_2 / "model_time_improved.keras")
}
ultra_modes = joblib.load(MODEL_DIR_2 / "models_ultra.pkl")
scaler_list = joblib.load(MODEL_DIR_2 / ["scaler_minmax.pkl", "scaler_ultra.pkl", "scaler.pkl"][0])
pickup_location = joblib.load(MODEL_DIR_2 / "pickup_location_map.pkl")
route_list = joblib.load(MODEL_DIR_2 / ["route_hour_dict_ctat.pkl", "route_hour_dict_vtat.pkl"][0])

app = FastAPI(title="Ride Prediction API", version="1.0")

# Define logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ride_api")

# Function to load all models and see what is inside each models