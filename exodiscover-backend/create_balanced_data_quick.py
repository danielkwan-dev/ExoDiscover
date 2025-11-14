"""
Quick solution: Create balanced training data by augmenting existing data
"""
import numpy as np
from scipy import signal

print("Creating balanced training data from existing samples...")

# Load current data
windows = np.load('training_data/windows.npy')
labels = np.load('training_data/labels.npy')

print(f"Current data: {len(windows)} windows")
print(f"  - Exoplanets: {sum(labels == 1)}")
print(f"  - Non-exoplanets: {sum(labels == 0)}")

# Separate classes
exo_windows = windows[labels == 1]
non_exo_windows = windows[labels == 0]

# Augment non-exoplanet data by adding noise and slight variations
augmented_non_exo = []
for window in non_exo_windows:
    # Original
    augmented_non_exo.append(window)

    # Add 15 variations with different noise/shifts
    for i in range(15):
        # Add small gaussian noise
        noise = np.random.normal(0, 0.0001, len(window))
        noisy = window + noise
        augmented_non_exo.append(noisy)

augmented_non_exo = np.array(augmented_non_exo)

# Randomly sample exoplanets to balance
num_samples = len(augmented_non_exo)
exo_indices = np.random.choice(len(exo_windows), size=num_samples, replace=True)
balanced_exo = exo_windows[exo_indices]

# Combine
balanced_windows = np.concatenate([balanced_exo, augmented_non_exo])
balanced_labels = np.concatenate([
    np.ones(len(balanced_exo)),
    np.zeros(len(augmented_non_exo))
])

# Shuffle
shuffle_idx = np.random.permutation(len(balanced_windows))
balanced_windows = balanced_windows[shuffle_idx]
balanced_labels = balanced_labels[shuffle_idx]

print(f"\nBalanced data created: {len(balanced_windows)} windows")
print(f"  - Exoplanets: {sum(balanced_labels == 1)}")
print(f"  - Non-exoplanets: {sum(balanced_labels == 0)}")

# Save
np.save('training_data/windows.npy', balanced_windows)
np.save('training_data/labels.npy', balanced_labels)

print("\nSaved balanced data to training_data/")
print("Now run: python train_cnn.py")
