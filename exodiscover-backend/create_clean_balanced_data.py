"""
Create CLEAN balanced training data by filtering bad samples
"""
import pandas as pd
import numpy as np
from scipy import signal

print("Creating CLEAN balanced training data...")
print("=" * 60)

# Load exoTest.csv
exoTest = pd.read_csv('test_data/exoTest.csv')
flux_cols = [col for col in exoTest.columns if col.startswith('FLUX')]

windows = []
labels = []

print(f"Loaded exoTest.csv: {len(exoTest)} samples")

# Process NON-CANDIDATES (false positives)
non_candidates = exoTest[exoTest['LABEL'] == 1]
print(f"\nProcessing {len(non_candidates)} non-candidates...")

count = 0
for idx, row in non_candidates.iterrows():
    if count >= 100:  # Limit to 100
        break

    flux = row[flux_cols].values.astype(float)

    # Skip if has NaN, zeros, or infinities
    if np.any(np.isnan(flux)) or np.any(np.isinf(flux)) or np.any(flux == 0):
        continue

    # Normalize safely
    median_val = np.median(flux)
    if median_val == 0:
        continue

    flux = flux / median_val

    # Check again after normalization
    if np.any(np.isnan(flux)) or np.any(np.isinf(flux)):
        continue

    # Resample to 256
    if len(flux) != 256:
        flux = signal.resample(flux, 256)

    windows.append(flux)
    labels.append(0)  # Not exoplanet
    count += 1

print(f"  Added {count} clean non-exoplanet samples")

# Load original exoplanet data (which is clean)
print(f"\nLoading original exoplanet data...")
original_windows = np.load('training_data/windows.npy')
original_labels = np.load('training_data/labels.npy')

# Get only exoplanet samples from original data
exo_windows_original = original_windows[original_labels == 1]
print(f"  Found {len(exo_windows_original)} exoplanet windows in original data")

# Take same number as non-exoplanets for balance
num_to_take = min(count, len(exo_windows_original))
selected_exo = exo_windows_original[np.random.choice(len(exo_windows_original), num_to_take, replace=False)]

for window in selected_exo:
    windows.append(window)
    labels.append(1)

print(f"  Added {num_to_take} exoplanet samples")

# Convert to arrays
windows = np.array(windows)
labels = np.array(labels)

# Shuffle
shuffle_idx = np.random.permutation(len(windows))
windows = windows[shuffle_idx]
labels = labels[shuffle_idx]

print("\n" + "=" * 60)
print("CLEAN BALANCED DATASET CREATED")
print("=" * 60)
print(f"Total windows: {len(windows)}")
print(f"  - Exoplanet (label=1): {sum(labels == 1)} ({sum(labels==1)/len(labels)*100:.1f}%)")
print(f"  - Not Exoplanet (label=0): {sum(labels == 0)} ({sum(labels==0)/len(labels)*100:.1f}%)")

# Verify no NaN/Inf
has_nan = np.any(np.isnan(windows))
has_inf = np.any(np.isinf(windows))
print(f"\nData quality check:")
print(f"  Contains NaN: {has_nan}")
print(f"  Contains Inf: {has_inf}")

if not has_nan and not has_inf:
    # Save
    np.save('training_data/windows.npy', windows)
    np.save('training_data/labels.npy', labels)
    print(f"\nSaved clean data to training_data/")
    print("\nNext: Run 'python train_cnn.py'")
else:
    print("\nERROR: Data still contains NaN or Inf! Not saving.")
