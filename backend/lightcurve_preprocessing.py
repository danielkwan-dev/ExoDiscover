import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create output directory for plots and preprocessed data
Path("lightcurve_data").mkdir(exist_ok=True)

print("Downloading sample Kepler light curve data...")
# Download a known exoplanet transit: Kepler-10b (one of the first confirmed rocky exoplanets)
search_result = lk.search_lightcurve('Kepler-10', author='Kepler', cadence='long')
print(f"Found {len(search_result)} light curve files for Kepler-10")

# Download ALL quarters and stitch them together
lc = search_result.download_all().stitch()
print(f"\nLight curve downloaded: {len(lc.time)} data points")
print(f"Time range: {lc.time.min().value:.2f} to {lc.time.max().value:.2f} BKJD")

# ============================
# 1. PLOT RAW LIGHT CURVE
# ============================
plt.figure(figsize=(12, 4))
plt.plot(lc.time.value, lc.flux.value, 'k.', markersize=1)
plt.xlabel('Time (BKJD)')
plt.ylabel('Flux (electrons/s)')
plt.title('Raw Kepler-10 Light Curve')
plt.tight_layout()
plt.savefig('lightcurve_data/01_raw_lightcurve.png', dpi=150)
print("\n✓ Saved: lightcurve_data/01_raw_lightcurve.png")
plt.close()

# ============================
# 2. PREPROCESSING: NORMALIZE
# ============================
print("\n--- Preprocessing Step 1: Normalization ---")
# Remove NaN values
lc_clean = lc.remove_nans()
print(f"Removed {len(lc) - len(lc_clean)} NaN values")

# Normalize flux to median = 1
lc_normalized = lc_clean.normalize()
print(f"Normalized flux: median = {np.median(lc_normalized.flux.value):.4f}")

plt.figure(figsize=(12, 4))
plt.plot(lc_normalized.time.value, lc_normalized.flux.value, 'k.', markersize=1)
plt.xlabel('Time (BKJD)')
plt.ylabel('Normalized Flux')
plt.title('Normalized Light Curve (median = 1)')
plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Median')
plt.legend()
plt.tight_layout()
plt.savefig('lightcurve_data/02_normalized_lightcurve.png', dpi=150)
print("✓ Saved: lightcurve_data/02_normalized_lightcurve.png")
plt.close()

# ============================
# 3. PREPROCESSING: HANDLE GAPS
# ============================
print("\n--- Preprocessing Step 2: Handle Gaps/Missing Data ---")
# Identify time gaps (cadence for Kepler long mode ~30 min = 0.02 days)
time_diff = np.diff(lc_normalized.time.value)
gap_threshold = 0.1  # days
gaps = np.where(time_diff > gap_threshold)[0]
print(f"Found {len(gaps)} time gaps > {gap_threshold} days")

# Flatten the light curve to remove long-term trends (stellar variability)
lc_flattened = lc_normalized.flatten(window_length=401)
print("Applied Savitzky-Golay filter to remove stellar variability")

plt.figure(figsize=(12, 4))
plt.plot(lc_flattened.time.value, lc_flattened.flux.value, 'k.', markersize=1)
plt.xlabel('Time (BKJD)')
plt.ylabel('Flattened Flux')
plt.title('Flattened Light Curve (trends removed)')
plt.tight_layout()
plt.savefig('lightcurve_data/03_flattened_lightcurve.png', dpi=150)
print("✓ Saved: lightcurve_data/03_flattened_lightcurve.png")
plt.close()

# ============================
# 4. PREPROCESSING: SEGMENT INTO WINDOWS
# ============================
print("\n--- Preprocessing Step 3: Segment into Windows ---")
# For CNN input, we need fixed-length windows
# Using smaller window size to ensure we get enough windows
window_size = 256
stride = 128  # 50% overlap

flux_data = lc_flattened.flux.value
time_data = lc_flattened.time.value

# Create sliding windows
windows = []
window_times = []
for i in range(0, len(flux_data) - window_size + 1, stride):
    window = flux_data[i:i + window_size]
    window_time = time_data[i:i + window_size]

    # Only keep windows without major gaps
    if np.all(np.diff(window_time) < gap_threshold):
        windows.append(window)
        window_times.append(window_time)

windows = np.array(windows)
print(f"Created {len(windows)} windows of size {window_size}")
print(f"Window shape: {windows.shape}")

# Plot first 3 windows
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for i in range(min(3, len(windows))):
    axes[i].plot(window_times[i], windows[i], 'k-', linewidth=0.5)
    axes[i].set_xlabel('Time (BKJD)')
    axes[i].set_ylabel('Flux')
    axes[i].set_title(f'Window {i+1}')
    axes[i].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lightcurve_data/04_segmented_windows.png', dpi=150)
print("✓ Saved: lightcurve_data/04_segmented_windows.png")
plt.close()

# ============================
# 5. SAVE PREPROCESSED DATA
# ============================
print("\n--- Saving Preprocessed Arrays ---")

# Save first 2 windows as numpy arrays for testing (if we have any)
if len(windows) > 0:
    np.save('lightcurve_data/window_1.npy', windows[0])
    print(f"✓ Saved: lightcurve_data/window_1.npy (shape: {windows[0].shape})")

    if len(windows) > 1:
        np.save('lightcurve_data/window_2.npy', windows[1])
        print(f"✓ Saved: lightcurve_data/window_2.npy (shape: {windows[1].shape})")

    # Save full preprocessed dataset
    np.save('lightcurve_data/all_windows.npy', windows)
    print(f"✓ Saved: lightcurve_data/all_windows.npy (shape: {windows.shape})")
else:
    print("⚠ No windows created - dataset too small or too many gaps")

# Save metadata
metadata = {
    'target': 'Kepler-10',
    'window_size': window_size,
    'stride': stride,
    'num_windows': len(windows),
    'time_range': (float(lc.time.min().value), float(lc.time.max().value)),
    'preprocessing_steps': [
        'Remove NaNs',
        'Normalize to median=1',
        'Flatten (remove trends)',
        'Segment into fixed windows'
    ]
}
np.save('lightcurve_data/metadata.npy', metadata, allow_pickle=True)
print("✓ Saved: lightcurve_data/metadata.npy")

print("\n" + "="*50)
print("PREPROCESSING SUMMARY")
print("="*50)
print(f"Target: Kepler-10 (confirmed exoplanet host)")
print(f"Original data points: {len(lc)}")
print(f"After cleaning: {len(lc_clean)}")
print(f"Segmented windows: {len(windows)}")
print(f"Window size: {window_size} points")
print(f"\nPreprocessing steps applied:")
for i, step in enumerate(metadata['preprocessing_steps'], 1):
    print(f"  {i}. {step}")
print(f"\nAll data saved to: lightcurve_data/")
print("="*50)
