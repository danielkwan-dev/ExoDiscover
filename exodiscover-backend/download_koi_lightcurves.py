"""
Download real light curves from the KOI dataset
This will download actual Kepler light curves for CONFIRMED, CANDIDATE, and FALSE POSITIVE
"""
import numpy as np
import pandas as pd
import lightkurve as lk
from tqdm import tqdm
import os

print("="*70)
print("DOWNLOADING REAL KOI LIGHT CURVES")
print("="*70)

# Load KOI dataset
df = pd.read_csv('model/lighkurve_KOI_dataset.csv')

print(f"\nKOI Dataset: {len(df)} records")
print(f"  CONFIRMED: {sum(df['koi_disposition'] == 'CONFIRMED')}")
print(f"  CANDIDATE: {sum(df['koi_disposition'] == 'CANDIDATE')}")
print(f"  FALSE POSITIVE: {sum(df['koi_disposition'] == 'FALSE POSITIVE')}")

# Map dispositions to numeric labels
label_map = {
    'CONFIRMED': 2,
    'CANDIDATE': 1,
    'FALSE POSITIVE': 0
}

def download_lightcurves_for_class(disposition, target_windows=200, max_targets=20):
    """Download light curves for a specific disposition class"""
    # Get unique KEPIDs for this disposition
    kepids = df[df['koi_disposition'] == disposition]['kepid'].unique()

    print(f"\n{'='*70}")
    print(f"DOWNLOADING {disposition} (Label {label_map[disposition]})")
    print(f"{'='*70}")
    print(f"Available KEPIDs: {len(kepids)}")
    print(f"Target: {target_windows} windows from up to {max_targets} targets")

    all_windows = []
    all_labels = []
    label = label_map[disposition]

    successful = 0

    for kepid in tqdm(kepids[:max_targets], desc=f"Downloading {disposition}"):
        if len(all_windows) >= target_windows:
            break

        try:
            # Search by KEPID
            search = lk.search_lightcurve(f'KIC {kepid}', author='Kepler', cadence='long')

            if len(search) == 0:
                continue

            # Download all available quarters
            lc_collection = search.download_all()
            if lc_collection is None:
                continue

            # Stitch quarters together
            if hasattr(lc_collection, 'stitch'):
                lc = lc_collection.stitch()
            else:
                lc = lc_collection

            # Get flux
            flux = lc.flux.value
            flux = flux[~np.isnan(flux)]

            if len(flux) < 256:
                continue

            # Normalize
            flux = flux / np.median(flux)

            # Segment into 256-point windows
            windows_added = 0
            for i in range(0, len(flux) - 256, 128):  # 50% overlap
                if len(all_windows) >= target_windows:
                    break

                window = flux[i:i+256]

                if len(window) == 256 and not np.any(np.isnan(window)) and not np.any(np.isinf(window)):
                    all_windows.append(window)
                    all_labels.append(label)
                    windows_added += 1

            if windows_added > 0:
                successful += 1

        except Exception as e:
            continue

    print(f"Result: {len(all_windows)} windows from {successful} KEPIDs")
    return all_windows, all_labels

# Download data for each class
WINDOWS_PER_CLASS = 200
MAX_TARGETS = 30

all_windows = []
all_labels = []

# Download CONFIRMED
windows, labels = download_lightcurves_for_class('CONFIRMED', WINDOWS_PER_CLASS, MAX_TARGETS)
all_windows.extend(windows)
all_labels.extend(labels)

# Download CANDIDATE
windows, labels = download_lightcurves_for_class('CANDIDATE', WINDOWS_PER_CLASS, MAX_TARGETS)
all_windows.extend(windows)
all_labels.extend(labels)

# Download FALSE POSITIVE
windows, labels = download_lightcurves_for_class('FALSE POSITIVE', WINDOWS_PER_CLASS, MAX_TARGETS)
all_windows.extend(windows)
all_labels.extend(labels)

print("\n" + "="*70)
print("SAVING TRAINING DATA")
print("="*70)
print(f"Total windows: {len(all_windows)}")
print(f"  Confirmed (2): {sum(1 for l in all_labels if l == 2)}")
print(f"  Candidate (1): {sum(1 for l in all_labels if l == 1)}")
print(f"  False Positive (0): {sum(1 for l in all_labels if l == 0)}")

if len(all_windows) > 0:
    # Shuffle
    indices = np.random.permutation(len(all_windows))
    all_windows = np.array(all_windows)[indices]
    all_labels = np.array(all_labels)[indices]

    os.makedirs('training_data', exist_ok=True)
    np.save('training_data/windows.npy', all_windows)
    np.save('training_data/labels.npy', all_labels)

    print("\nSaved to training_data/")
    print("="*70)
else:
    print("\nERROR: No data downloaded!")
    print("="*70)
