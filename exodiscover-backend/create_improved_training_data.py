"""
Create IMPROVED training data with CLEAR distinguishing features:
- Exoplanets: Light curves WITH transit dips (periodic drops in brightness)
- Non-exoplanets: Smooth curves WITHOUT transit dips
"""
import numpy as np
from scipy import signal

print("Creating IMPROVED training dataset with clear features...")
print("=" * 60)

num_samples = 200  # 200 each class for better training

# ============================================================================
# EXOPLANET LIGHT CURVES - WITH CLEAR TRANSIT DIPS
# ============================================================================
print(f"\nCreating {num_samples} EXOPLANET light curves (WITH transits)...")

exoplanet_windows = []

for i in range(num_samples):
    # Base stellar flux (normalized to 1.0)
    base_flux = np.ones(256)

    # Add small stellar noise
    noise = np.random.normal(0, 0.0003, 256)

    # Add stellar variability (slow changes)
    stellar_var = 0.0005 * np.sin(2 * np.pi * np.arange(256) / 40)

    # CREATE TRANSIT DIP - this is the KEY feature!
    transit_depth = np.random.uniform(0.005, 0.02)  # 0.5% to 2% dip
    transit_duration = int(np.random.uniform(8, 20))  # Transit lasts 8-20 points

    # Random transit position (not too close to edges)
    transit_start = np.random.randint(30, 256 - transit_duration - 30)

    # Create smooth transit shape (Gaussian-like dip)
    transit_shape = np.zeros(256)
    transit_x = np.arange(transit_duration)
    transit_center = transit_duration / 2
    # Gaussian dip
    transit_profile = transit_depth * np.exp(-((transit_x - transit_center)**2) / (transit_duration/3)**2)
    transit_shape[transit_start:transit_start + transit_duration] = transit_profile

    # Combine all components (flux DECREASES during transit)
    lightcurve = base_flux + noise + stellar_var - transit_shape

    exoplanet_windows.append(lightcurve)

exoplanet_windows = np.array(exoplanet_windows)
print(f"  Created {len(exoplanet_windows)} exoplanet samples")
print(f"  Range: {exoplanet_windows.min():.6f} to {exoplanet_windows.max():.6f}")

# ============================================================================
# NON-EXOPLANET LIGHT CURVES - NO TRANSITS (smooth)
# ============================================================================
print(f"\nCreating {num_samples} NON-EXOPLANET light curves (NO transits)...")

non_exoplanet_windows = []

for i in range(num_samples):
    # Base flux
    base_flux = np.ones(256)

    # Small noise
    noise = np.random.normal(0, 0.0003, 256)

    # Stellar variability (can be stronger than exoplanet cases)
    period = np.random.uniform(20, 60)
    amplitude = np.random.uniform(0.0005, 0.002)
    stellar_var = amplitude * np.sin(2 * np.pi * np.arange(256) / period)

    # Slow drift
    drift = np.linspace(-0.001, 0.001, 256) * np.random.choice([-1, 0, 1])

    # NO TRANSIT DIPS - just smooth variations
    lightcurve = base_flux + noise + stellar_var + drift

    non_exoplanet_windows.append(lightcurve)

non_exoplanet_windows = np.array(non_exoplanet_windows)
print(f"  Created {len(non_exoplanet_windows)} non-exoplanet samples")
print(f"  Range: {non_exoplanet_windows.min():.6f} to {non_exoplanet_windows.max():.6f}")

# ============================================================================
# COMBINE AND SHUFFLE
# ============================================================================
all_windows = np.concatenate([exoplanet_windows, non_exoplanet_windows])
all_labels = np.concatenate([
    np.ones(len(exoplanet_windows)),       # 1 = Exoplanet
    np.zeros(len(non_exoplanet_windows))   # 0 = Not Exoplanet
])

# Shuffle
shuffle_idx = np.random.permutation(len(all_windows))
all_windows = all_windows[shuffle_idx]
all_labels = all_labels[shuffle_idx]

print("\n" + "=" * 60)
print("IMPROVED DATASET CREATED")
print("=" * 60)
print(f"Total windows: {len(all_windows)}")
print(f"  - Exoplanet (label=1): {sum(all_labels == 1)} ({sum(all_labels==1)/len(all_labels)*100:.1f}%)")
print(f"  - Not Exoplanet (label=0): {sum(all_labels == 0)} ({sum(all_labels==0)/len(all_labels)*100:.1f}%)")

# Quality check
has_nan = np.any(np.isnan(all_windows))
has_inf = np.any(np.isinf(all_windows))
print(f"\nData quality:")
print(f"  Contains NaN: {has_nan}")
print(f"  Contains Inf: {has_inf}")

if not has_nan and not has_inf:
    # Save
    np.save('training_data/windows.npy', all_windows)
    np.save('training_data/labels.npy', all_labels)
    print(f"\nSaved to training_data/")
    print("\nKEY DIFFERENCE:")
    print("  - Exoplanets have CLEAR transit dips (0.5-2% drops)")
    print("  - Non-exoplanets have NO dips (only smooth variations)")
    print("\nNext: Run 'python train_cnn.py'")
else:
    print("\nERROR: Data quality check failed!")
