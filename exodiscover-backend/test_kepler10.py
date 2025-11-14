"""
Test the lightcurve model on Kepler-10 data
Kepler-10 is a CONFIRMED exoplanet, so it should predict "Confirmed"
"""
import numpy as np
import joblib
from scipy import signal, stats

print("="*80)
print("TESTING MODEL ON KEPLER-10 (CONFIRMED EXOPLANET)")
print("="*80)

# Load the model
model = joblib.load('model/lightcurve_xgb_model.pkl')

# Load Kepler-10 light curve windows
kepler10_windows = np.load('lightcurve_data/all_windows.npy')

print(f"\nKepler-10 data: {len(kepler10_windows)} windows")
print("Kepler-10 is a CONFIRMED exoplanet")
print("Expected prediction: 'Confirmed' for most/all windows")

# Extract features function (same as in predict.py)
def extract_features(lightcurve):
    features = {}
    features['mean'] = np.mean(lightcurve)
    features['std'] = np.std(lightcurve)
    features['median'] = np.median(lightcurve)
    features['max'] = np.max(lightcurve)
    features['min'] = np.min(lightcurve)
    features['range'] = features['max'] - features['min']
    features['coefficient_variation'] = features['std'] / features['mean'] if features['mean'] != 0 else 0

    median_flux = features['median']
    dips = lightcurve < (median_flux - 2 * features['std'])
    features['num_dips'] = np.sum(dips)
    features['dip_fraction'] = features['num_dips'] / len(lightcurve)
    features['max_dip_depth'] = median_flux - np.min(lightcurve[dips]) if features['num_dips'] > 0 else 0

    features['skewness'] = stats.skew(lightcurve)
    features['kurtosis'] = stats.kurtosis(lightcurve)

    autocorr = np.correlate(lightcurve - np.mean(lightcurve), lightcurve - np.mean(lightcurve), mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    if len(autocorr) > 10:
        peaks, _ = signal.find_peaks(autocorr[5:], height=0)
        features['num_periods'] = len(peaks)
        features['primary_period'] = peaks[0] + 5 if len(peaks) > 0 else 0
    else:
        features['num_periods'] = 0
        features['primary_period'] = 0

    diff = np.diff(lightcurve)
    noise_estimate = np.std(diff) / np.sqrt(2)
    features['snr'] = features['max_dip_depth'] / noise_estimate if noise_estimate > 0 else 0
    features['smoothness'] = np.mean(np.abs(np.diff(lightcurve)))

    threshold = median_flux - 1.5 * features['std']
    below_threshold = lightcurve < threshold
    transit_count = 0
    in_transit = False
    transit_lengths = []
    current_length = 0

    for point in below_threshold:
        if point:
            if not in_transit:
                transit_count += 1
                in_transit = True
            current_length += 1
        else:
            if in_transit:
                transit_lengths.append(current_length)
                current_length = 0
            in_transit = False

    features['transit_count'] = transit_count
    features['avg_transit_length'] = np.mean(transit_lengths) if transit_lengths else 0
    features['max_transit_length'] = np.max(transit_lengths) if transit_lengths else 0

    return features

# Feature names (must match training order)
feature_names = ['mean', 'std', 'median', 'max', 'min', 'range', 'coefficient_variation',
                'num_dips', 'dip_fraction', 'max_dip_depth', 'skewness', 'kurtosis',
                'num_periods', 'primary_period', 'snr', 'smoothness', 'transit_count',
                'avg_transit_length', 'max_transit_length']

# Test on 20 windows
print("\n" + "="*80)
print("TESTING 20 KEPLER-10 WINDOWS")
print("="*80)

label_map = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}
predictions = []

print("\nResults:")
for i in range(min(20, len(kepler10_windows))):
    window = kepler10_windows[i]

    # Extract features
    features = extract_features(window)
    feature_array = np.array([[features[name] for name in feature_names]])

    # Predict
    pred = model.predict(feature_array)[0]
    prob = model.predict_proba(feature_array)[0]
    conf = prob[int(pred)]

    pred_label = label_map[int(pred)]
    predictions.append(int(pred))

    status = "CORRECT" if pred_label == "Confirmed" else "WRONG"
    print(f"  Window {i:3d}: {pred_label:15s} ({conf:.1%}) [{status}]")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

confirmed_count = predictions.count(2)
candidate_count = predictions.count(1)
false_pos_count = predictions.count(0)

print(f"Predictions:")
print(f"  Confirmed: {confirmed_count}/20 ({confirmed_count/20*100:.1f}%)")
print(f"  Candidate: {candidate_count}/20 ({candidate_count/20*100:.1f}%)")
print(f"  False Positive: {false_pos_count}/20 ({false_pos_count/20*100:.1f}%)")

print(f"\nExpected: Mostly 'Confirmed' (Kepler-10 is a confirmed exoplanet)")

if confirmed_count >= 15:
    print("Result: EXCELLENT - Model correctly identifies Kepler-10!")
elif confirmed_count >= 10:
    print("Result: GOOD - Model mostly identifies Kepler-10 correctly")
else:
    print("Result: NEEDS IMPROVEMENT - Model struggles with Kepler-10")

print("="*80)
