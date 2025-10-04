from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load preprocessed light curve data at startup
LIGHTCURVE_DATA_DIR = 'lightcurve_data'
preprocessed_windows = None
metadata = None

if os.path.exists(os.path.join(LIGHTCURVE_DATA_DIR, 'all_windows.npy')):
    preprocessed_windows = np.load(os.path.join(LIGHTCURVE_DATA_DIR, 'all_windows.npy'))
    metadata = np.load(os.path.join(LIGHTCURVE_DATA_DIR, 'metadata.npy'), allow_pickle=True).item()
    print(f"Loaded {len(preprocessed_windows)} preprocessed light curve windows")
else:
    print("Warning: No preprocessed light curve data found. Run lightcurve_preprocessing.py first.")

@app.route('/predict_tabular', methods=['POST'])
def predict_tabular():
    """Predict exoplanet classification from tabular data"""
    return jsonify({
        "label": "Exoplanet",
        "confidence": 0.93
    })

@app.route('/predict_lightcurve', methods=['POST'])
def predict_lightcurve():
    """Predict exoplanet candidate from light curve data"""

    # Get request data (optional: allow user to specify which window)
    try:
        data = request.get_json(silent=True) or {}
    except:
        data = {}
    window_index = data.get('window_index', 0)

    # Use real preprocessed data if available
    if preprocessed_windows is not None and len(preprocessed_windows) > 0:
        # Get a specific window or default to first one
        window_index = min(window_index, len(preprocessed_windows) - 1)
        lightcurve_data = preprocessed_windows[window_index].tolist()

        # Dummy prediction (replace with actual model later)
        confidence = 0.77 + (window_index * 0.01)  # Vary confidence slightly
        label = "Candidate" if confidence < 0.85 else "Exoplanet"

        response = {
            "label": label,
            "confidence": round(confidence, 2),
            "lightcurve": lightcurve_data,
            "metadata": {
                "window_index": window_index,
                "total_windows": len(preprocessed_windows),
                "target": metadata.get('target', 'Unknown') if metadata else 'Unknown',
                "preprocessing_applied": metadata.get('preprocessing_steps', []) if metadata else []
            }
        }
    else:
        # Fallback to dummy data if preprocessing hasn't been run
        dummy_lightcurve = [
            0.998, 0.999, 1.000, 0.999, 0.998, 0.997, 0.995, 0.993,
            0.991, 0.990, 0.989, 0.988, 0.988, 0.989, 0.990, 0.991,
            0.993, 0.995, 0.997, 0.998, 0.999, 1.000, 0.999, 0.998
        ]
        response = {
            "label": "Candidate",
            "confidence": 0.77,
            "lightcurve": dummy_lightcurve,
            "metadata": {
                "warning": "Using dummy data. Run lightcurve_preprocessing.py to use real data."
            }
        }

    return jsonify(response)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
