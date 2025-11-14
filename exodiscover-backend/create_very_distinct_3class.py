"""
Create VERY DISTINCT 3-class data so model can easily tell them apart:
- FALSE POSITIVE (0): Smooth sine waves, NO sharp dips
- CANDIDATE (1): ONE medium dip (1-2%)
- CONFIRMED (2): MULTIPLE deep dips (3-5%)
"""
import numpy as np
import os

np.random.seed(42)

print("="*70)
print("CREATING VERY DISTINCT 3-CLASS DATA")
print("="*70)

def create_false_positive():
    """Class 0: SMOOTH sine waves, NO dips at all"""
    base = np.ones(256)

    # Multiple smooth sine waves (NO sharp features)
    freq1 = np.random.uniform(20, 50)
    freq2 = np.random.uniform(60, 100)
    amp1 = np.random.uniform(0.005, 0.015)
    amp2 = np.random.uniform(0.003, 0.008)

    x = np.arange(256)
    wave1 = amp1 * np.sin(2 * np.pi * x / freq1)
    wave2 = amp2 * np.sin(2 * np.pi * x / freq2)

    # Small noise
    noise = np.random.normal(0, 0.0005, 256)

    # Gradual trend
    trend = np.linspace(np.random.uniform(-0.005, 0), np.random.uniform(0, 0.005), 256)

    lightcurve = base + wave1 + wave2 + noise + trend
    return np.clip(lightcurve, 0.98, 1.02)

def create_candidate():
    """Class 1: ONE medium dip (1-2%)"""
    base = np.ones(256)

    # Light noise
    noise = np.random.normal(0, 0.0002, 256)

    # ONE MEDIUM DIP
    dip_depth = np.random.uniform(0.01, 0.02)  # 1-2%
    dip_width = int(np.random.uniform(12, 18))
    dip_center = np.random.randint(60, 196)

    x = np.arange(256)
    dip = np.zeros(256)
    for i in range(256):
        if abs(i - dip_center) < dip_width:
            distance = abs(i - dip_center)
            dip[i] = dip_depth * np.exp(-(distance**2) / (dip_width/2.5)**2)

    # Small stellar variation
    stellar = 0.003 * np.sin(2 * np.pi * x / 70)

    lightcurve = base - dip + noise + stellar
    return np.clip(lightcurve, 0.97, 1.005)

def create_confirmed():
    """Class 2: MULTIPLE deep dips (3-5%)"""
    base = np.ones(256)

    # Very low noise (high quality data)
    noise = np.random.normal(0, 0.0001, 256)

    # 2-3 DEEP DIPS
    num_dips = np.random.randint(2, 4)

    total_dip = np.zeros(256)
    spacing = 256 // (num_dips + 1)

    for j in range(num_dips):
        dip_depth = np.random.uniform(0.03, 0.05)  # 3-5% DEEP!
        dip_width = int(np.random.uniform(15, 25))
        dip_center = spacing * (j + 1) + np.random.randint(-20, 20)

        if dip_center < 30 or dip_center > 226:
            continue

        x = np.arange(256)
        for i in range(256):
            if abs(i - dip_center) < dip_width:
                distance = abs(i - dip_center)
                total_dip[i] += dip_depth * np.exp(-(distance**2) / (dip_width/3)**2)

    lightcurve = base - total_dip + noise
    return np.clip(lightcurve, 0.945, 1.0)

# Generate balanced dataset
samples_per_class = 300

all_windows = []
all_labels = []

print(f"\nGenerating {samples_per_class} samples per class...")
print("- FALSE POSITIVE: Smooth waves, NO dips")
print("- CANDIDATE: ONE medium dip (1-2%)")
print("- CONFIRMED: MULTIPLE deep dips (3-5%)")

# False Positive
print("\nGenerating False Positives...")
for i in range(samples_per_class):
    all_windows.append(create_false_positive())
    all_labels.append(0)

# Candidate
print("Generating Candidates...")
for i in range(samples_per_class):
    all_windows.append(create_candidate())
    all_labels.append(1)

# Confirmed
print("Generating Confirmed...")
for i in range(samples_per_class):
    all_windows.append(create_confirmed())
    all_labels.append(2)

# Shuffle
indices = np.random.permutation(len(all_windows))
all_windows = np.array(all_windows)[indices]
all_labels = np.array(all_labels)[indices]

print("\n" + "="*70)
print("SAVING TRAINING DATA")
print("="*70)
print(f"Total windows: {len(all_windows)}")
print(f"  Confirmed (2): {sum(all_labels == 2)} - MULTIPLE deep dips (3-5%)")
print(f"  Candidate (1): {sum(all_labels == 1)} - ONE medium dip (1-2%)")
print(f"  False Positive (0): {sum(all_labels == 0)} - Smooth, NO dips")

os.makedirs('training_data', exist_ok=True)
np.save('training_data/windows.npy', all_windows)
np.save('training_data/labels.npy', all_labels)

print("\nSaved to training_data/")
print("="*70)
