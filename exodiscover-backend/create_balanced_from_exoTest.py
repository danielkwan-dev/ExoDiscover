"""
Create balanced training data by combining original data with exoTest.csv
Original: 214 exoplanets, 11 non-exoplanets (too many exo)
exoTest: 5 exoplanets, 565 non-exoplanets (too many non-exo)
Combined: Should be more balanced
"""
import pandas as pd
import numpy as np
from scipy import signal

print("Creating balanced training data...")
print("=" * 60)

# Load exoTest.csv
exoTest = pd.read_csv('test_data/exoTest.csv')
print(f"Loaded exoTest.csv: {len(exoTest)} samples")
print(f"  - CANDIDATE (label=2): {sum(exoTest['LABEL'] == 2)}")
print(f"  - NON-CANDIDATE (label=1): {sum(exoTest['LABEL'] == 1)}")

# Convert FLUX columns to windows
flux_cols = [col for col in exoTest.columns if col.startswith('FLUX')]
print(f"Found {len(flux_cols)} flux measurements per sample")

windows = []
labels = []

# Take 100 non-candidates (false positives) from exoTest
non_candidates = exoTest[exoTest['LABEL'] == 1].head(100)
for idx, row in non_candidates.iterrows():
    flux = row[flux_cols].values.astype(float)
    # Normalize
    flux = flux / np.median(flux)
    # Resample to 256 points
    if len(flux) != 256:
        flux = signal.resample(flux, 256)
    windows.append(flux)
    labels.append(0)  # Not exoplanet

print(f"Added {len(non_candidates)} non-exoplanet samples from exoTest")

# Take all 5 candidates from exoTest
candidates = exoTest[exoTest['LABEL'] == 2]
for idx, row in candidates.iterrows():
    flux = row[flux_cols].values.astype(float)
    flux = flux / np.median(flux)
    if len(flux) != 256:
        flux = signal.resample(flux, 256)
    windows.append(flux)
    labels.append(1)  # Exoplanet

# Load original exoplanet windows and add 95 more
original_windows = np.load('training_data/windows.npy')
original_labels = np.load('training_data/labels.npy')

exo_windows_original = original_windows[original_labels == 1]
# Take 95 exoplanet samples to balance with 100 non-exo
selected_exo = exo_windows_original[np.random.choice(len(exo_windows_original), 95, replace=False)]

for window in selected_exo:
    windows.append(window)
    labels.append(1)

# Convert to arrays
windows = np.array(windows)
labels = np.array(labels)

# Shuffle
shuffle_idx = np.random.permutation(len(windows))
windows = windows[shuffle_idx]
labels = labels[shuffle_idx]

print("\n" + "=" * 60)
print("BALANCED DATASET CREATED")
print("=" * 60)
print(f"Total windows: {len(windows)}")
print(f"  - Exoplanet (label=1): {sum(labels == 1)} ({sum(labels==1)/len(labels)*100:.1f}%)")
print(f"  - Not Exoplanet (label=0): {sum(labels == 0)} ({sum(labels==0)/len(labels)*100:.1f}%)")

# Save
np.save('training_data/windows.npy', windows)
np.save('training_data/labels.npy', labels)
print(f"\nSaved to training_data/")
print("\nNext: Run 'python train_cnn.py'")
