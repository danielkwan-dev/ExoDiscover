"""
Extract FEATURES from light curves instead of using raw data
Features astronomers use to identify exoplanets:
- Transit depth (how deep the dip is)
- Transit duration (how long the dip lasts)
- Number of transits detected
- Periodicity (are dips regular?)
- Signal-to-noise ratio
- Variability metrics
"""
import numpy as np
import pandas as pd
from scipy import signal, stats
from tqdm import tqdm

print("="*70)
print("EXTRACTING FEATURES FROM LIGHT CURVES")
print("="*70)

# Load the real KOI light curve data
windows = np.load('training_data/windows.npy')
labels = np.load('training_data/labels.npy')

print(f"\nLoaded {len(windows)} light curve windows")
print(f"  Confirmed (2): {sum(labels==2)}")
print(f"  Candidate (1): {sum(labels==1)}")
print(f"  False Positive (0): {sum(labels==0)}")

def extract_features(lightcurve):
    """Extract astronomical features from a light curve"""
    features = {}

    # 1. Basic statistics
    features['mean'] = np.mean(lightcurve)
    features['std'] = np.std(lightcurve)
    features['median'] = np.median(lightcurve)
    features['max'] = np.max(lightcurve)
    features['min'] = np.min(lightcurve)

    # 2. Variability
    features['range'] = features['max'] - features['min']
    features['coefficient_variation'] = features['std'] / features['mean'] if features['mean'] != 0 else 0

    # 3. Detect dips (values below median)
    median_flux = features['median']
    dips = lightcurve < (median_flux - 2 * features['std'])  # Points more than 2 sigma below median
    features['num_dips'] = np.sum(dips)
    features['dip_fraction'] = features['num_dips'] / len(lightcurve)

    # 4. Maximum dip depth
    if features['num_dips'] > 0:
        features['max_dip_depth'] = median_flux - np.min(lightcurve[dips])
    else:
        features['max_dip_depth'] = 0

    # 5. Skewness and Kurtosis (shape of distribution)
    features['skewness'] = stats.skew(lightcurve)
    features['kurtosis'] = stats.kurtosis(lightcurve)

    # 6. Detect periodic signals
    # Use autocorrelation to find periodicity
    autocorr = np.correlate(lightcurve - np.mean(lightcurve),
                           lightcurve - np.mean(lightcurve),
                           mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    # Find peaks in autocorrelation (excluding the first peak at lag 0)
    if len(autocorr) > 10:
        peaks, _ = signal.find_peaks(autocorr[5:], height=0)
        features['num_periods'] = len(peaks)
        if len(peaks) > 0:
            features['primary_period'] = peaks[0] + 5  # Add back the offset
        else:
            features['primary_period'] = 0
    else:
        features['num_periods'] = 0
        features['primary_period'] = 0

    # 7. Signal-to-noise ratio
    # Estimate noise from high-frequency components
    diff = np.diff(lightcurve)
    noise_estimate = np.std(diff) / np.sqrt(2)
    signal_strength = features['max_dip_depth']
    features['snr'] = signal_strength / noise_estimate if noise_estimate > 0 else 0

    # 8. Smoothness (how smooth is the curve?)
    features['smoothness'] = np.mean(np.abs(np.diff(lightcurve)))

    # 9. Count significant dips (potential transits)
    # Look for consecutive points below threshold
    threshold = median_flux - 1.5 * features['std']
    below_threshold = lightcurve < threshold

    # Count groups of consecutive points
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

print("\nExtracting features from all light curves...")
all_features = []

for i in tqdm(range(len(windows)), desc="Extracting features"):
    features = extract_features(windows[i])
    all_features.append(features)

# Convert to DataFrame
features_df = pd.DataFrame(all_features)
features_df['label'] = labels

print("\n" + "="*70)
print("FEATURE EXTRACTION COMPLETE")
print("="*70)
print(f"\nExtracted {len(features_df.columns)-1} features:")
print(list(features_df.columns[:-1]))

print(f"\nSample statistics by class:")
for label in [0, 1, 2]:
    label_name = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}[label]
    subset = features_df[features_df['label'] == label]
    print(f"\n{label_name}:")
    print(f"  Mean dip depth: {subset['max_dip_depth'].mean():.6f}")
    print(f"  Mean transit count: {subset['transit_count'].mean():.2f}")
    print(f"  Mean SNR: {subset['snr'].mean():.2f}")

# Save features
features_df.to_csv('training_data/lightcurve_features.csv', index=False)
print(f"\nSaved features to training_data/lightcurve_features.csv")
print("="*70)
