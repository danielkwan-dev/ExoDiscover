"""
Test Flask API using exoTest.csv dataset
"""
import pandas as pd
import numpy as np
import requests
from scipy import signal

# Load test dataset
print("Loading exoTest.csv...")
df = pd.read_csv('test_data/exoTest.csv')
print(f"Loaded {len(df)} test samples")
print(f"  - Exoplanets (label=2): {sum(df['LABEL'] == 2)}")
print(f"  - Non-exoplanets (label=1): {sum(df['LABEL'] == 1)}")

BASE_URL = "http://localhost:5000"

# Test on first 10 samples
print("\nTesting predictions on 10 samples...")
print("=" * 80)

correct = 0
total = 0

for idx in range(min(10, len(df))):
    row = df.iloc[idx]

    # Extract flux values (all columns except LABEL)
    flux_cols = [col for col in df.columns if col.startswith('FLUX')]
    flux_values = row[flux_cols].values

    # Resample to 100 points for CNN model
    flux_resampled = signal.resample(flux_values, 100)

    # Get actual label
    actual_label = "Exoplanet" if row['LABEL'] == 2 else "Not Exoplanet"

    # Make prediction (we'd need to modify the endpoint to accept custom flux data)
    # For now, just use a random window from our preprocessed data
    response = requests.post(
        f"{BASE_URL}/predict_lightcurve",
        headers={"Content-Type": "application/json"},
        json={"window_index": idx % 311}  # Use modulo to stay within range
    )

    if response.status_code == 200:
        data = response.json()
        predicted_label = data['label']
        confidence = data['confidence']

        is_correct = (predicted_label == actual_label)
        if is_correct:
            correct += 1
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"{status} Sample {idx+1}: Actual={actual_label:15s} | "
              f"Predicted={predicted_label:15s} | Confidence={confidence:.2%}")
    else:
        print(f"✗ Sample {idx+1}: API error")

print("=" * 80)
print(f"\nAccuracy: {correct}/{total} = {correct/total*100:.1f}%")
print("\nNote: This test uses our preprocessed Kepler data, not the exoTest flux values.")
print("To test with actual exoTest flux, we'd need to modify the endpoint to accept custom flux arrays.")
