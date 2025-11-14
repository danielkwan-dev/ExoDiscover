"""
Create synthetic 3-class training data for CNN:
- Class 0: False Positive (noise, no clear transit pattern)
- Class 1: Candidate (weak/uncertain transit-like dips)
- Class 2: Confirmed Exoplanet (strong, clear transit dips)
"""
import numpy as np
import os

np.random.seed(42)

print("="*70)
print("CREATING SYNTHETIC 3-CLASS TRAINING DATA")
print("="*70)

def create_confirmed_exoplanet():
    """Class 2: Strong, clear transit dips"""
    base_flux = np.ones(256)
    noise = np.random.normal(0, 0.0002, 256)

    # Stellar variability
    period = np.random.uniform(40, 80)
    amplitude = np.random.uniform(0.0005, 0.0015)
    stellar_var = amplitude * np.sin(2 * np.pi * np.arange(256) / period)

    # STRONG CLEAR TRANSIT DIP (2-5%)
    transit_depth = np.random.uniform(0.02, 0.05)
    transit_duration = int(np.random.uniform(10, 20))
    transit_start = np.random.randint(30, 256 - transit_duration - 30)

    transit_x = np.arange(transit_duration)
    transit_center = transit_duration / 2
    transit_profile = transit_depth * np.exp(-((transit_x - transit_center)**2) / (transit_duration/3)**2)

    transit_shape = np.zeros(256)
    transit_shape[transit_start:transit_start+transit_duration] = transit_profile

    lightcurve = base_flux + noise + stellar_var - transit_shape
    lightcurve = np.clip(lightcurve, 0.95, 1.005)

    return lightcurve

def create_candidate():
    """Class 1: Weaker, less certain transit-like features"""
    base_flux = np.ones(256)
    noise = np.random.normal(0, 0.0004, 256)  # More noise than confirmed

    # Stellar variability
    period = np.random.uniform(40, 80)
    amplitude = np.random.uniform(0.001, 0.003)  # More variability
    stellar_var = amplitude * np.sin(2 * np.pi * np.arange(256) / period)

    # WEAK/UNCERTAIN TRANSIT-LIKE DIP (0.5-2%)
    transit_depth = np.random.uniform(0.005, 0.02)
    transit_duration = int(np.random.uniform(8, 15))
    transit_start = np.random.randint(40, 256 - transit_duration - 40)

    transit_x = np.arange(transit_duration)
    transit_center = transit_duration / 2
    # Less smooth profile
    transit_profile = transit_depth * np.exp(-((transit_x - transit_center)**2) / (transit_duration/4)**2)

    transit_shape = np.zeros(256)
    transit_shape[transit_start:transit_start+transit_duration] = transit_profile

    # Add some irregularity
    irregularity = np.random.normal(0, 0.0005, 256)

    lightcurve = base_flux + noise + stellar_var - transit_shape + irregularity
    lightcurve = np.clip(lightcurve, 0.985, 1.01)

    return lightcurve

def create_false_positive():
    """Class 0: No transit, just noise and stellar variability"""
    base_flux = np.ones(256)
    noise = np.random.normal(0, 0.0005, 256)  # High noise

    # Strong stellar variability (mimics but isn't transit)
    period = np.random.uniform(30, 100)
    amplitude = np.random.uniform(0.002, 0.005)
    stellar_var = amplitude * np.sin(2 * np.pi * np.arange(256) / period)

    # Random drift
    drift_direction = np.random.choice([-1, 0, 1])
    drift = np.linspace(-0.001, 0.001, 256) * drift_direction

    # Maybe add some random bumps (not transit-shaped)
    if np.random.random() > 0.5:
        bump_center = np.random.randint(50, 200)
        bump_width = np.random.randint(30, 60)
        x = np.arange(256)
        bump = 0.003 * np.exp(-((x - bump_center)**2) / (bump_width**2))
        lightcurve = base_flux + noise + stellar_var + drift + bump
    else:
        lightcurve = base_flux + noise + stellar_var + drift

    lightcurve = np.clip(lightcurve, 0.99, 1.01)

    return lightcurve

# Generate balanced dataset
samples_per_class = 200
total = samples_per_class * 3

all_windows = []
all_labels = []

print(f"\nGenerating {samples_per_class} samples per class...")

# Class 2: Confirmed
print("Generating Confirmed Exoplanets (Class 2)...")
for i in range(samples_per_class):
    all_windows.append(create_confirmed_exoplanet())
    all_labels.append(2)

# Class 1: Candidate
print("Generating Candidates (Class 1)...")
for i in range(samples_per_class):
    all_windows.append(create_candidate())
    all_labels.append(1)

# Class 0: False Positive
print("Generating False Positives (Class 0)...")
for i in range(samples_per_class):
    all_windows.append(create_false_positive())
    all_labels.append(0)

# Shuffle
indices = np.random.permutation(total)
all_windows = np.array(all_windows)[indices]
all_labels = np.array(all_labels)[indices]

print("\n" + "="*70)
print("SAVING TRAINING DATA")
print("="*70)
print(f"Total windows: {len(all_windows)}")
print(f"  Confirmed (label=2): {sum(all_labels == 2)}")
print(f"  Candidate (label=1): {sum(all_labels == 1)}")
print(f"  False Positive (label=0): {sum(all_labels == 0)}")

os.makedirs('training_data', exist_ok=True)

np.save('training_data/windows.npy', all_windows)
np.save('training_data/labels.npy', all_labels)

print("\nSaved to training_data/")
print("="*70)
