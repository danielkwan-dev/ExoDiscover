"""
Create highly distinguishable 3-class training data:
- Class 0 (False Positive): Random noise, NO dips at all
- Class 1 (Candidate): 1 small dip (0.5-1.5%)
- Class 2 (Confirmed): Multiple strong dips (2-4%)
"""
import numpy as np
import os

np.random.seed(42)

print("="*70)
print("CREATING DISTINGUISHABLE 3-CLASS TRAINING DATA")
print("="*70)

def create_confirmed():
    """Class 2: Multiple STRONG, DEEP transit dips (2-4%)"""
    base = np.ones(256)
    noise = np.random.normal(0, 0.0001, 256)  # Low noise

    # Create 2-3 transit dips
    num_transits = np.random.randint(2, 4)

    for _ in range(num_transits):
        depth = np.random.uniform(0.02, 0.04)  # DEEP: 2-4%
        duration = int(np.random.uniform(15, 25))
        start = np.random.randint(20, 256 - duration - 20)

        # Sharp, clear dip
        for i in range(duration):
            base[start + i] -= depth * np.exp(-((i - duration/2)**2) / (duration/4)**2)

    lightcurve = base + noise
    return np.clip(lightcurve, 0.96, 1.0)

def create_candidate():
    """Class 1: ONE weak, shallow dip (0.5-1.5%)"""
    base = np.ones(256)
    noise = np.random.normal(0, 0.0003, 256)  # More noise

    # Single weak dip
    depth = np.random.uniform(0.005, 0.015)  # SHALLOW: 0.5-1.5%
    duration = int(np.random.uniform(10, 18))
    start = np.random.randint(40, 256 - duration - 40)

    # Less pronounced dip
    for i in range(duration):
        base[start + i] -= depth * np.exp(-((i - duration/2)**2) / (duration/5)**2)

    # Add some stellar variability
    stellar = 0.002 * np.sin(2 * np.pi * np.arange(256) / 60)

    lightcurve = base + noise + stellar
    return np.clip(lightcurve, 0.985, 1.005)

def create_false_positive():
    """Class 0: NO dips, just smooth curves and noise"""
    base = np.ones(256)
    noise = np.random.normal(0, 0.0004, 256)  # High noise

    # Strong stellar variability (sinusoidal, NOT dips)
    period = np.random.uniform(30, 100)
    amplitude = np.random.uniform(0.003, 0.007)
    stellar = amplitude * np.sin(2 * np.pi * np.arange(256) / period)

    # Random drift
    drift = np.linspace(np.random.uniform(-0.003, 0), np.random.uniform(0, 0.003), 256)

    # Maybe add gradual trend (NOT sharp dip)
    if np.random.random() > 0.5:
        trend = 0.005 * np.sin(2 * np.pi * np.arange(256) / 120)
    else:
        trend = 0

    lightcurve = base + noise + stellar + drift + trend
    return np.clip(lightcurve, 0.99, 1.01)

# Generate more samples per class
samples_per_class = 300
total = samples_per_class * 3

all_windows = []
all_labels = []

print(f"\nGenerating {samples_per_class} samples per class...")

# Class 2: Confirmed (multiple deep dips)
print("Generating CONFIRMED: Multiple strong dips (2-4% depth)...")
for i in range(samples_per_class):
    all_windows.append(create_confirmed())
    all_labels.append(2)

# Class 1: Candidate (single weak dip)
print("Generating CANDIDATES: One weak dip (0.5-1.5% depth)...")
for i in range(samples_per_class):
    all_windows.append(create_candidate())
    all_labels.append(1)

# Class 0: False Positive (no dips)
print("Generating FALSE POSITIVES: No dips, smooth variation...")
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
print(f"  Confirmed (label=2): {sum(all_labels == 2)} - Multiple deep dips")
print(f"  Candidate (label=1): {sum(all_labels == 1)} - One weak dip")
print(f"  False Positive (label=0): {sum(all_labels == 0)} - No dips")

os.makedirs('training_data', exist_ok=True)

np.save('training_data/windows.npy', all_windows)
np.save('training_data/labels.npy', all_labels)

print("\nSaved to training_data/")
print("="*70)
