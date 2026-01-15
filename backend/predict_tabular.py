import joblib
import numpy as np
import pandas as pd

# Load Wen's trained XGBoost model
try:
    model = joblib.load("model/xgb_wen_model.pkl")
    MODEL_LOADED = True
    print("Loaded Wen's XGBoost model")
except FileNotFoundError as e:
    print(f"Warning: Could not load XGBoost model: {e}")
    MODEL_LOADED = False

def predict_from_features(features_dict):
    """
    Predict exoplanet classification from tabular features

    Args:
        features_dict: Dictionary with keys matching the feature names

    Returns:
        dict with 'label' and 'confidence'
    """
    if not MODEL_LOADED:
        return {"label": "Unknown", "confidence": 0.5, "error": "Model not loaded"}

    # Extract features in the correct order (match Wen's training order)
    # Wen's model expects 7 features
    feature_names = [
        'orbital_period',
        'transit_duration',
        'transit_depth',
        'planet_radius',
        'eq_temperature',
        'stellar_temp',
        'stellar_radius'
    ]

    # Build feature array, handling missing values
    features = []
    for feat in feature_names:
        val = features_dict.get(feat)
        if val is None or pd.isna(val):
            val = 0  # Use 0 for missing features (or could use median)
        features.append(float(val))

    features_array = np.array([features])

    # Make prediction
    try:
        prediction = model.predict(features_array)[0]
        probabilities = model.predict_proba(features_array)[0]
        confidence = float(max(probabilities))

        # Map prediction to label (0 = Not Exoplanet, 1 = Exoplanet)
        label = "Confirmed Exoplanet" if int(prediction) == 1 else "False Positive"

        return {
            "label": label,
            "confidence": round(confidence, 3)
        }
    except Exception as e:
        return {
            "label": "Error",
            "confidence": 0.0,
            "error": str(e)
        }
