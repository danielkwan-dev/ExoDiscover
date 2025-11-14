"""
Create final balanced training data using:
- Clean Kepler-10 exoplanet light curves (150 samples)
- Realistic synthetic non-exoplanet light curves (150 samples)
"""
import numpy as np

print("Creating FINAL balanced training dataset...")
print("=" * 60)

# Load clean Kepler-10 data (confirmed exoplanet)
kepler_windows = np.load('lightcurve_data/all_windows.npy')
print(f"Loaded {len(kepler_windows)} Kepler-10 windows (exoplanet)")
print(f"  Shape: {kepler_windows.shape}")
print(f"  Range: {kepler_windows.min():.6f} to {kepler_windows.max():.6f}")

# Take 150 exoplanet samples
num_samples = 150
exo_samples = kepler_windows[np.random.choice(len(kepler_windows), num_samples, replace=False)]

# Create realistic synthetic NON-exoplanet light curves
# Non-exoplanets have:
# - NO transit dips (smooth curves)
# - Small random variations from stellar activity
# - Possible slow trends from instrumental effects
print(f"\nGenerating {num_samples} synthetic non-exoplanet light curves...")

non_exo_samples = []

for i in range(num_samples):
    # Base: normalized around 1.0 with small random walk
    noise = np.random.normal(0, 0.00015, 256)  # Very small noise
    cumulative_noise = np.cumsum(noise)

    # Add slight linear trend (instrumental drift)
    trend = np.linspace(-0.0005, 0.0005, 256) * np.random.choice([-1, 0, 1])

    # Add periodic stellar variability (rotation)
    period = np.random.uniform(10, 50)  # Random period
    amplitude = np.random.uniform(0.0002, 0.0008)
    stellar_var = amplitude * np.sin(2 * np.pi * np.arange(256) / period)

    # Combine all components
    lightcurve = 1.0 + cumulative_noise + trend + stellar_var

    # Normalize to have same range as Kepler data
    lightcurve = lightcurve - np.median(lightcurve) + 1.0

    non_exo_samples.append(lightcurve)

non_exo_samples = np.array(non_exo_samples)

print(f"  Generated samples range: {non_exo_samples.min():.6f} to {non_exo_samples.max():.6f}")

# Combine datasets
all_windows = np.concatenate([exo_samples, non_exo_samples])
all_labels = np.concatenate([
    np.ones(len(exo_samples)),      # 1 = Exoplanet
    np.zeros(len(non_exo_samples))  # 0 = Not Exoplanet
])

# Shuffle
shuffle_idx = np.random.permutation(len(all_windows))
all_windows = all_windows[shuffle_idx]
all_labels = all_labels[shuffle_idx]

print("\n" + "=" * 60)
print("FINAL BALANCED DATASET")
print("=" * 60)
print(f"Total windows: {len(all_windows)}")
print(f"  - Exoplanet (label=1): {sum(all_labels == 1)} ({sum(all_labels==1)/len(all_labels)*100:.1f}%)")
print(f"  - Not Exoplanet (label=0): {sum(all_labels == 0)} ({sum(all_labels==0)/len(all_labels)*100:.1f}%)")

# Quality check
has_nan = np.any(np.isnan(all_windows))
has_inf = np.any(np.isinf(all_windows))
print(f"\nData quality check:")
print(f"  Contains NaN: {has_nan}")
print(f"  Contains Inf: {has_inf}")
print(f"  Data range: {all_windows.min():.6f} to {all_windows.max():.6f}")

if not has_nan and not has_inf:
    # Save
    np.save('training_data/windows.npy', all_windows)
    np.save('training_data/labels.npy', all_labels)
    print(f"\n✓ Saved clean balanced data to training_data/")
    print("\nNext: Run 'python train_cnn.py'")
else:
    print("\nERROR: Data quality check failed!")
