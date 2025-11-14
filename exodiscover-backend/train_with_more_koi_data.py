"""
Download and train with MORE KOI data for better model diversity
Target: 500+ windows per class from multiple different KEPIDs
"""
import numpy as np
import pandas as pd
import lightkurve as lk
from tqdm import tqdm
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy import signal, stats
import os

print("="*70)
print("DOWNLOADING MORE KOI DATA FOR IMPROVED MODEL")
print("="*70)

# Load KOI dataset
df = pd.read_csv('model/lighkurve_KOI_dataset.csv')

print(f"\nKOI Dataset: {len(df)} records")
print(f"  CONFIRMED: {sum(df['koi_disposition'] == 'CONFIRMED')}")
print(f"  CANDIDATE: {sum(df['koi_disposition'] == 'CANDIDATE')}")
print(f"  FALSE POSITIVE: {sum(df['koi_disposition'] == 'FALSE POSITIVE')}")

label_map = {'CONFIRMED': 2, 'CANDIDATE': 1, 'FALSE POSITIVE': 0}

def download_for_class(disposition, target_windows=500, max_kepids=50):
    """Download light curves for a specific disposition class"""
    kepids = df[df['koi_disposition'] == disposition]['kepid'].unique()

    print(f"\n{'='*70}")
    print(f"DOWNLOADING {disposition} (Label {label_map[disposition]})")
    print(f"{'='*70}")
    print(f"Available KEPIDs: {len(kepids)}")
    print(f"Target: {target_windows} windows from up to {max_kepids} KEPIDs")

    all_windows = []
    all_labels = []
    label = label_map[disposition]
    successful = 0

    for kepid in tqdm(kepids[:max_kepids], desc=f"{disposition}"):
        if len(all_windows) >= target_windows:
            break

        try:
            search = lk.search_lightcurve(f'KIC {kepid}', author='Kepler', cadence='long')
            if len(search) == 0:
                continue

            lc_collection = search.download_all()
            if lc_collection is None:
                continue

            if hasattr(lc_collection, 'stitch'):
                lc = lc_collection.stitch()
            else:
                lc = lc_collection

            flux = lc.flux.value
            flux = flux[~np.isnan(flux)]

            if len(flux) < 256:
                continue

            flux = flux / np.median(flux)

            # Take windows with 50% overlap
            for i in range(0, len(flux) - 256, 128):
                if len(all_windows) >= target_windows:
                    break

                window = flux[i:i+256]
                if len(window) == 256 and not np.any(np.isnan(window)) and not np.any(np.isinf(window)):
                    all_windows.append(window)
                    all_labels.append(label)

            if len(all_windows) > len(all_labels) - 1:
                successful += 1

        except Exception:
            continue

    print(f"Result: {len(all_windows)} windows from {successful} KEPIDs")
    return all_windows, all_labels

# Download 500 windows per class
WINDOWS_PER_CLASS = 500
MAX_KEPIDS = 50

all_windows = []
all_labels = []

# Download each class
for disposition in ['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE']:
    windows, labels = download_for_class(disposition, WINDOWS_PER_CLASS, MAX_KEPIDS)
    all_windows.extend(windows)
    all_labels.extend(labels)

print("\n" + "="*70)
print("DOWNLOADED DATA SUMMARY")
print("="*70)
print(f"Total windows: {len(all_windows)}")
print(f"  Confirmed (2): {sum(1 for l in all_labels if l == 2)}")
print(f"  Candidate (1): {sum(1 for l in all_labels if l == 1)}")
print(f"  False Positive (0): {sum(1 for l in all_labels if l == 0)}")

if len(all_windows) > 0:
    # Shuffle and save
    indices = np.random.permutation(len(all_windows))
    all_windows = np.array(all_windows)[indices]
    all_labels = np.array(all_labels)[indices]

    os.makedirs('training_data', exist_ok=True)
    np.save('training_data/windows.npy', all_windows)
    np.save('training_data/labels.npy', all_labels)

    print("\nSaved raw windows to training_data/")

    # Extract features and train
    print("\n" + "="*70)
    print("EXTRACTING FEATURES")
    print("="*70)

    def extract_features(lightcurve):
        """Extract astronomical features"""
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
    for window in tqdm(all_windows, desc="Extracting features"):
        all_features.append(extract_features(window))

    features_df = pd.DataFrame(all_features)
    features_df['label'] = all_labels
    features_df.to_csv('training_data/lightcurve_features.csv', index=False)

    print(f"Extracted {len(features_df.columns)-1} features")

    # Train model
    print("\n" + "="*70)
    print("TRAINING XGBOOST MODEL")
    print("="*70)

    X = features_df.drop('label', axis=1)
    y = features_df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    print(f"\nTest Accuracy: {test_accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['False Positive', 'Candidate', 'Confirmed']))

    joblib.dump(model, 'model/lightcurve_xgb_model.pkl')
    print("\nModel saved to model/lightcurve_xgb_model.pkl")
    print("="*70)

else:
    print("\nERROR: No data downloaded!")
