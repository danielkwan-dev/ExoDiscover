"""
Download balanced light curve dataset:
- 50 confirmed exoplanets
- 50 false positives (non-exoplanets)
"""
import lightkurve as lk
import numpy as np
import pandas as pd
from pathlib import Path

Path("training_data").mkdir(exist_ok=True)

print("Downloading BALANCED light curve training data from NASA...")
print("=" * 60)

# Load Kepler data with dispositions
kepler_df = pd.read_csv('nasa_data/kepler.csv', comment='#')

# Get confirmed planets
confirmed = kepler_df[kepler_df['koi_disposition'] == 'CONFIRMED']
print(f"Found {len(confirmed)} confirmed exoplanets")

# Get false positives
false_pos = kepler_df[kepler_df['koi_disposition'] == 'FALSE POSITIVE']
print(f"Found {len(false_pos)} false positives")

# Download function
def download_lightcurves(df, label, target_count=50):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    windows = []
    labels = []
    count = 0
    attempts = 0
    max_attempts = min(len(df), target_count * 5)  # Try up to 5x target

    label_name = "EXOPLANET" if label == 1 else "NON-EXOPLANET"
    print(f"\nDownloading {label_name} light curves...")
    print(f"Target: {target_count} systems")

    for idx, row in df.iterrows():
        if count >= target_count or attempts >= max_attempts:
            break

        attempts += 1

        # Get identifiers
        kepler_name = row.get('kepler_name')
        koi_name = row['kepoi_name']
        ra = row.get('ra')
        dec = row.get('dec')

        name_to_use = kepler_name if pd.notna(kepler_name) else koi_name

        try:
            print(f"  [{count+1}/{target_count}] {name_to_use}...", end=" ", flush=True)

            # Try name first (for confirmed planets with Kepler names)
            if pd.notna(kepler_name):
                search = lk.search_lightcurve(kepler_name, author='Kepler', cadence='long')
            # For false positives without Kepler names, use coordinates
            elif pd.notna(ra) and pd.notna(dec):
                coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
                search = lk.search_lightcurve(coord, radius=30*u.arcsec, author='Kepler', cadence='long')
            else:
                print("No valid identifier")
                continue

            if len(search) > 0:
                # Download all available quarters for this target
                lc_collection = search.download_all()

                if lc_collection is not None:
                    # Stitch quarters together if it's a collection
                    try:
                        lc = lc_collection.stitch()
                    except:
                        # If stitching fails or it's not a collection, use first item
                        if hasattr(lc_collection, '__getitem__'):
                            lc = lc_collection[0]
                        else:
                            lc = lc_collection

                    if lc is not None and len(lc.time) > 256:
                        # Preprocess
                        lc = lc.remove_nans().normalize().flatten(window_length=401)
                        flux = lc.flux.value

                        # Create only 1-2 windows per system (not too many)
                        window_size = 256
                        num_windows = min(2, (len(flux) - window_size) // window_size)

                        for i in range(num_windows):
                            start = i * window_size
                            window = flux[start:start + window_size]
                            windows.append(window)
                            labels.append(label)

                        count += 1
                        print(f"OK ({len(flux)} pts, {num_windows} windows)")
                    else:
                        print("Too short")
                else:
                    print("No data")
            else:
                print("Not found")
        except Exception as e:
            print(f"Error: {str(e)[:30]}")

    print(f"\nCompleted: {count}/{target_count} systems downloaded")
    return windows, labels

# Download balanced data (reduced count for faster download)
exo_windows, exo_labels = download_lightcurves(confirmed, label=1, target_count=30)
non_exo_windows, non_exo_labels = download_lightcurves(false_pos, label=0, target_count=30)

# Combine
all_windows = exo_windows + non_exo_windows
all_labels = exo_labels + non_exo_labels

print("\n" + "=" * 60)
print("BALANCED DATASET SUMMARY")
print("=" * 60)
print(f"Total windows: {len(all_windows)}")
print(f"  - Exoplanet (label=1): {sum(l == 1 for l in all_labels)}")
print(f"  - Not Exoplanet (label=0): {sum(l == 0 for l in all_labels)}")

# Save
np.save('training_data/windows.npy', np.array(all_windows))
np.save('training_data/labels.npy', np.array(all_labels))
print(f"\n✓ Saved to training_data/")
print("\nNext: Run 'python train_cnn.py' to train with balanced data")
