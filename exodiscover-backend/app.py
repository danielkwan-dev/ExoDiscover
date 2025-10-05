# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from predict import predict_lightcurve

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    df = pd.read_csv(file)  # must contain "flux" column
    flux = df["flux"].fillna(1.0).values[:2000]
    flux = flux / np.nanmedian(flux)

    result = predict_lightcurve(flux)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
