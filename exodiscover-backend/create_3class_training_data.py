"""
Create 3-class training data for CNN:
- Class 0: False Positive
- Class 1: Candidate
- Class 2: Confirmed Exoplanet
"""
import numpy as np
import pandas as pd
import lightkurve as lk
from scipy import signal
import os

print("="*70)
print("CREATING 3-CLASS TRAINING DATA")
print("="*70)

# Load the NASA data to find targets
df = pd.read_csv('merged_exoplanets.csv')

# Get targets for each class
confirmed = df[df['disposition'] == 2.0]['planet_name'].dropna().head(5).tolist()
candidates = df[df['disposition'] == 1.0]['planet_name'].dropna().head(5).tolist()
false_positives = df[df['disposition'] == 0.0]['planet_name'].dropna().head(5).tolist()

print(f"\nConfirmed targets: {confirmed}")
print(f"Candidate targets: {candidates}")
print(f"False positive targets: {false_positives}")

def download_and_segment(target_name, label, max_windows=100):
    """Download light curve and segment into windows"""
    try:
        print(f"\nDownloading {target_name} (label={label})...")
        search = lk.search_lightcurve(target_name, author='Kepler', cadence='long')

        if len(search) == 0:
            print(f"  No data found for {target_name}")
            return []

        lc = search.download_all()
        if lc is None:
            return []

        if hasattr(lc, 'stitch'):
            lc = lc.stitch()

        flux = lc.flux.value
        flux = flux[~np.isnan(flux)]

        if len(flux) < 256:
            return []

        # Normalize
        flux = flux / np.median(flux)

        # Segment into 256-point windows
        windows = []
        for i in range(0, len(flux) - 256, 128):
            window = flux[i:i+256]
            if len(window) == 256 and not np.any(np.isnan(window)):
                windows.append(window)
                if len(windows) >= max_windows:
                    break

        print(f"  Got {len(windows)} windows")
        return windows

    except Exception as e:
        print(f"  Error: {e}")
        return []

# Collect data for all 3 classes
all_windows = []
all_labels = []

# Download confirmed exoplanets (label=2)
print("\n" + "="*70)
print("DOWNLOADING CONFIRMED EXOPLANETS (Label 2)")
print("="*70)
for target in confirmed:
    windows = download_and_segment(target, label=2, max_windows=50)
    all_windows.extend(windows)
    all_labels.extend([2] * len(windows))

# Download candidates (label=1)
print("\n" + "="*70)
print("DOWNLOADING CANDIDATES (Label 1)")
print("="*70)
for target in candidates:
    windows = download_and_segment(target, label=1, max_windows=50)
    all_windows.extend(windows)
    all_labels.extend([1] * len(windows))

# Download false positives (label=0)
print("\n" + "="*70)
print("DOWNLOADING FALSE POSITIVES (Label 0)")
print("="*70)
for target in false_positives:
    windows = download_and_segment(target, label=0, max_windows=50)
    all_windows.extend(windows)
    all_labels.extend([0] * len(windows))

print("\n" + "="*70)
print("SAVING TRAINING DATA")
print("="*70)
print(f"Total windows: {len(all_windows)}")
print(f"  Confirmed (label=2): {sum(1 for l in all_labels if l == 2)}")
print(f"  Candidate (label=1): {sum(1 for l in all_labels if l == 1)}")
print(f"  False Positive (label=0): {sum(1 for l in all_labels if l == 0)}")

os.makedirs('training_data', exist_ok=True)

np.save('training_data/windows.npy', np.array(all_windows))
np.save('training_data/labels.npy', np.array(all_labels))

print("\nSaved to training_data/")
print("="*70)
