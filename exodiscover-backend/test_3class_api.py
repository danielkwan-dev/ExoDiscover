"""Test the 3-class lightcurve predictions via API"""
import requests
import time

print("="*70)
print("TESTING 3-CLASS LIGHTCURVE PREDICTIONS")
print("="*70)

# Wait for Flask to be ready
time.sleep(2)

# Test different windows to show variety in predictions
test_indices = [0, 5, 10, 15, 20, 50, 100, 150, 200, 250, 300]

labels_seen = set()

print("\nTesting predictions on different windows:")
print("-"*70)

for idx in test_indices:
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict_lightcurve",
            json={"window_index": idx},
            timeout=5
        )
        data = response.json()

        label = data.get('label', 'Unknown')
        confidence = data.get('confidence', 0)
        class_id = data.get('class_id', -1)

        labels_seen.add(label)

        print(f"Window {idx:3d}: {label:20s} (class {class_id}) - Confidence: {confidence:.2%}")

    except Exception as e:
        print(f"Window {idx:3d}: Error - {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Unique labels seen: {labels_seen}")
print("\nExpected labels:")
print("  - False Positive (class 0): No transit dips")
print("  - Candidate (class 1): Weak/single dip")
print("  - Confirmed (class 2): Multiple strong dips")
print("="*70)
