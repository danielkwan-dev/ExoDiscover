"""
Download real light curves from NASA for 3-class training:
- Confirmed exoplanets (disposition=2)
- Candidates (disposition=1)
- False positives (disposition=0)
"""
import numpy as np
import pandas as pd
import lightkurve as lk
from scipy import signal
import os
from tqdm import tqdm

print("="*70)
print("DOWNLOADING REAL 3-CLASS LIGHT CURVE DATA FROM NASA")
print("="*70)

# Load NASA data to get targets for each class
df = pd.read_csv('merged_exoplanets.csv')

# Get targets for each disposition class
confirmed_targets = df[df['disposition'] == 2.0]['planet_name'].dropna().unique().tolist()
candidate_targets = df[df['disposition'] == 1.0]['planet_name'].dropna().unique().tolist()
false_positive_targets = df[df['disposition'] == 0.0]['planet_name'].dropna().unique().tolist()

print(f"\nAvailable targets:")
print(f"  Confirmed: {len(confirmed_targets)}")
print(f"  Candidates: {len(candidate_targets)}")
print(f"  False Positives: {len(false_positive_targets)}")

def download_lightcurves(targets, label, label_name, target_windows=200, max_targets=20):
    """Download light curves and segment into windows"""
    all_windows = []
    all_labels = []

    print(f"\n{'='*70}")
    print(f"DOWNLOADING {label_name.upper()} (Label {label})")
    print(f"{'='*70}")
    print(f"Target: {target_windows} windows from up to {max_targets} targets")

    targets_tried = 0
    successful_targets = 0

    for target in tqdm(targets[:max_targets], desc=f"Downloading {label}"):
        if len(all_windows) >= target_windows:
            break

        targets_tried += 1

        try:
            # Search for Kepler long cadence data
            search = lk.search_lightcurve(target, author='Kepler', cadence='long')

            if len(search) == 0:
                continue

            # Download all available data
            lc_collection = search.download_all()
            if lc_collection is None:
                continue

            # Stitch together if multiple quarters
            if hasattr(lc_collection, 'stitch'):
                lc = lc_collection.stitch()
            else:
                lc = lc_collection

            # Get flux values
            flux = lc.flux.value

            # Remove NaNs
            flux = flux[~np.isnan(flux)]

            if len(flux) < 256:
                continue

            # Normalize to median
            flux = flux / np.median(flux)

            # Segment into 256-point windows with overlap
            windows_added = 0
            for i in range(0, len(flux) - 256, 128):  # 50% overlap
                window = flux[i:i+256]

                # Quality check
                if len(window) == 256 and not np.any(np.isnan(window)) and not np.any(np.isinf(window)):
                    all_windows.append(window)
                    all_labels.append(label)
                    windows_added += 1

                    if len(all_windows) >= target_windows:
                        break

            if windows_added > 0:
                successful_targets += 1

        except Exception as e:
            continue

    print(f"\nResults for {label_name}: {len(all_windows)} windows from {successful_targets}/{targets_tried} targets")
    return all_windows, all_labels

# Download data for each class
# We need balanced data, so same number for each class
WINDOWS_PER_CLASS = 200
MAX_TARGETS_PER_CLASS = 30

all_windows = []
all_labels = []

# Download Confirmed (label=2)
windows, labels = download_lightcurves(
    confirmed_targets,
    label=2,
    label_name="Confirmed",
    target_windows=WINDOWS_PER_CLASS,
    max_targets=MAX_TARGETS_PER_CLASS
)
all_windows.extend(windows)
all_labels.extend(labels)

# Download Candidates (label=1)
windows, labels = download_lightcurves(
    candidate_targets,
    label=1,
    label_name="Candidate",
    target_windows=WINDOWS_PER_CLASS,
    max_targets=MAX_TARGETS_PER_CLASS
)
all_windows.extend(windows)
all_labels.extend(labels)

# Download False Positives (label=0)
windows, labels = download_lightcurves(
    false_positive_targets,
    label=0,
    label_name="False Positive",
    target_windows=WINDOWS_PER_CLASS,
    max_targets=MAX_TARGETS_PER_CLASS
)
all_windows.extend(windows)
all_labels.extend(labels)

print("\n" + "="*70)
print("SAVING TRAINING DATA")
print("="*70)
print(f"Total windows: {len(all_windows)}")
print(f"  Confirmed (label=2): {sum(1 for l in all_labels if l == 2)}")
print(f"  Candidate (label=1): {sum(1 for l in all_labels if l == 1)}")
print(f"  False Positive (label=0): {sum(1 for l in all_labels if l == 0)}")

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
