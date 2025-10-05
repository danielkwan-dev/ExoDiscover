import lightkurve as lk
import numpy as np
import pandas as pd
from pathlib import Path

# Create directory for training data
Path("training_data").mkdir(exist_ok=True)

print("Downloading labeled light curve training data from NASA...")
print("=" * 60)

# Load the list of known exoplanets
exoplanets_df = pd.read_csv('../nasa_data/all_exoplanets_2021.csv')
print(f"Loaded {len(exoplanets_df)} confirmed exoplanets")

# Get Kepler planet names only (since we're using Kepler light curves)
kepler_planets = exoplanets_df[exoplanets_df['Planet Name'].str.contains('Kepler', case=False, na=False)]
print(f"Found {len(kepler_planets)} Kepler planets")

# Download light curves for confirmed exoplanets (label = 1)
exoplanet_windows = []
labels = []

print("\n1. Downloading CONFIRMED EXOPLANET light curves...")
count = 0
for idx, planet in kepler_planets.head(100).iterrows():  # Download up to 100 planets
    planet_name = planet['Planet Name']
    try:
        print(f"  Downloading {planet_name}...", end=" ")
        search = lk.search_lightcurve(planet_name, author='Kepler', cadence='long')

        if len(search) > 0:
            lc = search[0].download()
            if lc is not None and len(lc.time) > 0:
                # Preprocess
                lc = lc.remove_nans().normalize().flatten(window_length=401)

                # Segment into windows
                flux = lc.flux.value
                window_size = 256

                for i in range(0, len(flux) - window_size, window_size):
                    window = flux[i:i + window_size]
                    exoplanet_windows.append(window)
                    labels.append(1)  # Exoplanet

                count += 1
                print(f"✓ ({len(lc.flux)} points)")
            else:
                print("✗ No data")
        else:
            print("✗ Not found")
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")

    if count >= 50:  # Limit to 50 successful downloads
        break

print(f"\nCollected {len(exoplanet_windows)} exoplanet windows from {count} stars")

# Download light curves for NON-exoplanets (label = 0)
print("\n2. Downloading NON-EXOPLANET (false positive) light curves...")

# Search for Kepler false positives (KOIs marked as false positives)
non_exoplanet_windows = []
false_positive_kois = ['KOI-4', 'KOI-10', 'KOI-20', 'KOI-30', 'KOI-40',
                       'KOI-50', 'KOI-60', 'KOI-70', 'KOI-80', 'KOI-90']

count = 0
for koi_name in false_positive_kois:
    try:
        print(f"  Downloading {koi_name}...", end=" ")
        search = lk.search_lightcurve(koi_name, author='Kepler', cadence='long')

        if len(search) > 0:
            lc = search[0].download()
            if lc is not None and len(lc.time) > 0:
                # Preprocess
                lc = lc.remove_nans().normalize().flatten(window_length=401)

                # Segment into windows
                flux = lc.flux.value
                window_size = 256

                for i in range(0, len(flux) - window_size, window_size):
                    window = flux[i:i + window_size]
                    non_exoplanet_windows.append(window)
                    labels.append(0)  # Not exoplanet

                count += 1
                print(f"✓ ({len(lc.flux)} points)")
            else:
                print("✗ No data")
        else:
            print("✗ Not found")
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")

    if count >= 50:  # Limit to 50 successful downloads
        break

print(f"\nCollected {len(non_exoplanet_windows)} non-exoplanet windows from {count} stars")

# Combine all data
all_windows = exoplanet_windows + non_exoplanet_windows
all_labels = labels

print("\n" + "=" * 60)
print("TRAINING DATA SUMMARY")
print("=" * 60)
print(f"Total windows: {len(all_windows)}")
print(f"  - Exoplanet (label=1): {sum(l == 1 for l in all_labels)}")
print(f"  - Not Exoplanet (label=0): {sum(l == 0 for l in all_labels)}")

# Save training data
np.save('training_data/windows.npy', np.array(all_windows))
np.save('training_data/labels.npy', np.array(all_labels))
print(f"\n✓ Saved to training_data/windows.npy and training_data/labels.npy")
print("\nNext step: Run 'python train_cnn.py' to train the model")
