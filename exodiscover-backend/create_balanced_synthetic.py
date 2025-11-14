"""
Create balanced data using original exoplanets + synthetic non-exoplanets
"""
import numpy as np

print("Creating balanced data with synthetic non-exoplanets...")
print("=" * 60)

# Load original data
original_windows = np.load('training_data/windows.npy')
original_labels = np.load('training_data/labels.npy')

# Get exoplanet and non-exoplanet samples
exo_windows = original_windows[original_labels == 1]
non_exo_windows = original_windows[original_labels == 0]

print(f"Original data:")
print(f"  - {len(exo_windows)} exoplanet windows")
print(f"  - {len(non_exo_windows)} non-exoplanet windows")

# Take 100 exoplanet samples
num_samples = 100
exo_selected = exo_windows[np.random.choice(len(exo_windows), num_samples, replace=False)]

# Create synthetic non-exoplanet data
# Non-exoplanets have smoother, more stable light curves (less variability)
print(f"\nCreating {num_samples} synthetic non-exoplanet samples...")
synthetic_non_exo = []

for i in range(num_samples):
    # Start with a base of smooth noise
    base = np.random.normal(1.0, 0.0002, 256)  # Very small variations

    # Add slow drift
    drift = np.linspace(0, 0.001, 256) * np.random.choice([-1, 1])

    # Combine
    synthetic = base + drift

    # Ensure positive and normalized around 1
    synthetic = np.abs(synthetic)
    synthetic = synthetic / np.median(synthetic)

    synthetic_non_exo.append(synthetic)

synthetic_non_exo = np.array(synthetic_non_exo)

# Combine
all_windows = np.concatenate([exo_selected, synthetic_non_exo])
all_labels = np.concatenate([
    np.ones(len(exo_selected)),
    np.zeros(len(synthetic_non_exo))
])

# Shuffle
shuffle_idx = np.random.permutation(len(all_windows))
all_windows = all_windows[shuffle_idx]
all_labels = all_labels[shuffle_idx]

print("\n" + "=" * 60)
print("BALANCED DATASET CREATED")
print("=" * 60)
print(f"Total windows: {len(all_windows)}")
print(f"  - Exoplanet (label=1): {sum(all_labels == 1)} ({sum(all_labels==1)/len(all_labels)*100:.1f}%)")
print(f"  - Not Exoplanet (label=0): {sum(all_labels == 0)} ({sum(all_labels==0)/len(all_labels)*100:.1f}%)")

# Verify
has_nan = np.any(np.isnan(all_windows))
has_inf = np.any(np.isinf(all_windows))
print(f"\nData quality:")
print(f"  Contains NaN: {has_nan}")
print(f"  Contains Inf: {has_inf}")

# Save
np.save('training_data/windows.npy', all_windows)
np.save('training_data/labels.npy', all_labels)
print(f"\nSaved to training_data/")
print("\nNext: Run 'python train_cnn.py'")
