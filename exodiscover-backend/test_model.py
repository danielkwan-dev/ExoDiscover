from predict import predict_lightcurve

# dummy lightcurve of length 100
dummy_curve = [0.1 * i for i in range(100)]

prediction = predict_lightcurve(dummy_curve)
print("Prediction:", prediction)
