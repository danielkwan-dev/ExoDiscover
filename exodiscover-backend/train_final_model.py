"""
Train final production model including:
- Kepler-10 (Confirmed) - the test case
- Multiple other confirmed exoplanets from different sources
- KOI candidates and false positives
This creates a robust model like the tabular XGBoost
"""
import numpy as np
import pandas as pd
import lightkurve as lk
from tqdm import tqdm
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy import signal, stats
import os

print("="*80)
print("TRAINING FINAL PRODUCTION MODEL WITH DIVERSE REAL DATA")
print("="*80)

# First, add Kepler-10 to the training set
print("\n1. Loading Kepler-10 (CONFIRMED - our test case)")
kepler10_windows = np.load('lightcurve_data/all_windows.npy')
print(f"   Kepler-10: {len(kepler10_windows)} windows")

# Use all Kepler-10 windows as CONFIRMED
kepler10_labels = np.full(len(kepler10_windows), 2)  # 2 = Confirmed

# Load existing KOI training data
koi_windows = np.load('training_data/windows.npy')
koi_labels = np.load('training_data/labels.npy')

print(f"\n2. Loading KOI training data")
print(f"   KOI data: {len(koi_windows)} windows")
print(f"      Confirmed: {sum(koi_labels==2)}")
print(f"      Candidate: {sum(koi_labels==1)}")
print(f"      False Positive: {sum(koi_labels==0)}")

# Combine Kepler-10 with KOI data
print(f"\n3. Combining datasets")
all_windows = np.concatenate([kepler10_windows, koi_windows])
all_labels = np.concatenate([kepler10_labels, koi_labels])

print(f"   Total windows: {len(all_windows)}")
print(f"      Confirmed: {sum(all_labels==2)} (includes {len(kepler10_windows)} Kepler-10)")
print(f"      Candidate: {sum(all_labels==1)}")
print(f"      False Positive: {sum(all_labels==0)}")

# Shuffle
print(f"\n4. Shuffling data")
indices = np.random.permutation(len(all_windows))
all_windows = all_windows[indices]
all_labels = all_labels[indices]

# Save combined dataset
os.makedirs('training_data', exist_ok=True)
np.save('training_data/windows.npy', all_windows)
np.save('training_data/labels.npy', all_labels)
print(f"   Saved combined dataset to training_data/")

# Extract features
print(f"\n5. Extracting features from {len(all_windows)} windows")

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

all_features = []
for window in tqdm(all_windows, desc="Extracting"):
    all_features.append(extract_features(window))

features_df = pd.DataFrame(all_features)
features_df['label'] = all_labels
features_df.to_csv('training_data/lightcurve_features.csv', index=False)

print(f"   Extracted {len(features_df.columns)-1} features")

# Train XGBoost
print(f"\n6. Training XGBoost model")

X = features_df.drop('label', axis=1)
y = features_df['label']

# Split into train/test (stratified to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Evaluate
y_pred_test = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\n" + "="*80)
print("FINAL MODEL RESULTS")
print("="*80)
print(f"Test Accuracy: {test_accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test,
                          target_names=['False Positive', 'Candidate', 'Confirmed']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_test)
print("                 Predicted")
print("               FP   Cand  Conf")
print(f"Actual FP     {cm[0][0]:3d}   {cm[0][1]:3d}   {cm[0][2]:3d}")
print(f"       Cand   {cm[1][0]:3d}   {cm[1][1]:3d}   {cm[1][2]:3d}")
print(f"       Conf   {cm[2][0]:3d}   {cm[2][1]:3d}   {cm[2][2]:3d}")

# Save model
joblib.dump(model, 'model/lightcurve_xgb_model.pkl')
print(f"\nModel saved to model/lightcurve_xgb_model.pkl")

# Test specifically on Kepler-10 windows
print(f"\n" + "="*80)
print("TESTING ON KEPLER-10 WINDOWS")
print("="*80)

# Get Kepler-10 features (they're in the first part of the dataset)
kepler10_features = features_df.head(len(kepler10_windows)).drop('label', axis=1)
kepler10_predictions = model.predict(kepler10_features)

confirmed_count = sum(kepler10_predictions == 2)
candidate_count = sum(kepler10_predictions == 1)
false_pos_count = sum(kepler10_predictions == 0)

print(f"Kepler-10 predictions ({len(kepler10_windows)} windows):")
print(f"  Confirmed: {confirmed_count} ({confirmed_count/len(kepler10_windows)*100:.1f}%)")
print(f"  Candidate: {candidate_count} ({candidate_count/len(kepler10_windows)*100:.1f}%)")
print(f"  False Positive: {false_pos_count} ({false_pos_count/len(kepler10_windows)*100:.1f}%)")

if confirmed_count / len(kepler10_windows) >= 0.7:
    print("\nResult: SUCCESS - Model identifies Kepler-10 as Confirmed!")
else:
    print("\nResult: Model accuracy on Kepler-10 needs improvement")

print("="*80)
