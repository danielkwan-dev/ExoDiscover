from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
    # Dummy light curve array (time series data)
    dummy_lightcurve = [
        0.998, 0.999, 1.000, 0.999, 0.998, 0.997, 0.995, 0.993,
        0.991, 0.990, 0.989, 0.988, 0.988, 0.989, 0.990, 0.991,
        0.993, 0.995, 0.997, 0.998, 0.999, 1.000, 0.999, 0.998
    ]

    return jsonify({
        "label": "Candidate",
        "confidence": 0.77,
        "lightcurve": dummy_lightcurve
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
