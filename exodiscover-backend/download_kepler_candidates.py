"""
Quick download of Kepler CANDIDATE data to complete the 3-class dataset
We already have Confirmed and False Positive - just need Candidates
"""
import numpy as np
import pandas as pd
import lightkurve as lk
from tqdm import tqdm
import os

print("="*70)
print("DOWNLOADING KEPLER CANDIDATE DATA")
print("="*70)

# Load existing data
existing_windows = np.load('training_data/windows.npy')
existing_labels = np.load('training_data/labels.npy')

print(f"\nExisting data:")
print(f"  Confirmed (2): {sum(existing_labels==2)}")
print(f"  Candidate (1): {sum(existing_labels==1)}")
print(f"  False Positive (0): {sum(existing_labels==0)}")

# Load Kepler data with disposition=1 (Candidate)
df = pd.read_csv('merged_exoplanets.csv')
candidates = df[(df['disposition'] == 1.0) & (df['source'] == 'Kepler')]['planet_name'].dropna().unique().tolist()

print(f"\nKepler candidates available: {len(candidates)}")

# Download 200 windows
target_windows = 200
candidate_windows = []
candidate_labels = []

print(f"\nDownloading {target_windows} candidate windows...")

targets_tried = 0
for target in tqdm(candidates[:50], desc="Downloading"):  # Try up to 50 targets
    if len(candidate_windows) >= target_windows:
        break

    targets_tried += 1

    try:
        search = lk.search_lightcurve(target, author='Kepler', cadence='long')
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

        # Segment into windows
        for i in range(0, len(flux) - 256, 128):
            if len(candidate_windows) >= target_windows:
                break

            window = flux[i:i+256]
            if len(window) == 256 and not np.any(np.isnan(window)) and not np.any(np.isinf(window)):
                candidate_windows.append(window)
                candidate_labels.append(1)

    except Exception as e:
        continue

print(f"\nDownloaded {len(candidate_windows)} candidate windows from {targets_tried} targets")

if len(candidate_windows) > 0:
    # Combine with existing data
    all_windows = list(existing_windows) + candidate_windows
    all_labels = list(existing_labels) + candidate_labels

    # Shuffle
    indices = np.random.permutation(len(all_windows))
    all_windows = np.array(all_windows)[indices]
    all_labels = np.array(all_labels)[indices]

    # Save
    np.save('training_data/windows.npy', all_windows)
    np.save('training_data/labels.npy', all_labels)

    print("\n" + "="*70)
    print("UPDATED TRAINING DATA")
    print("="*70)
    print(f"Total windows: {len(all_windows)}")
    print(f"  Confirmed (2): {sum(all_labels==2)}")
    print(f"  Candidate (1): {sum(all_labels==1)}")
    print(f"  False Positive (0): {sum(all_labels==0)}")
    print("\nSaved to training_data/")
    print("="*70)
else:
    print("\nERROR: No candidate data downloaded!")
