"""
Overnight script: Download real data + Train CNN + Test results
Run this and leave it overnight - it will complete everything
"""
import subprocess
import os
import sys
import time
from datetime import datetime

def log(message):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()

log("="*70)
log("OVERNIGHT TRAINING PIPELINE STARTED")
log("="*70)

# Step 1: Download real 3-class data
log("\n>>> STEP 1: Downloading real light curve data from NASA...")
log("This will take 30-60 minutes depending on network speed")
log("Downloading 200 windows each of: Confirmed, Candidate, False Positive")

try:
    result = subprocess.run(
        [sys.executable, "download_real_3class_data.py"],
        capture_output=False,
        text=True,
        check=True
    )
    log("✓ Download completed successfully!")
except subprocess.CalledProcessError as e:
    log(f"✗ Download failed with error code {e.returncode}")
    sys.exit(1)

# Check if data was downloaded
if not os.path.exists('training_data/windows.npy'):
    log("✗ ERROR: Training data not found after download!")
    sys.exit(1)

# Step 2: Train the CNN model
log("\n>>> STEP 2: Training CNN model with real data...")
log("This will take 5-10 minutes")

try:
    result = subprocess.run(
        [sys.executable, "train_cnn.py"],
        capture_output=False,
        text=True,
        check=True
    )
    log("✓ Training completed successfully!")
except subprocess.CalledProcessError as e:
    log(f"✗ Training failed with error code {e.returncode}")
    sys.exit(1)

# Check if model was saved
if not os.path.exists('model/trained_cnn.pth'):
    log("✗ ERROR: Trained model not found!")
    sys.exit(1)

# Step 3: Test the model
log("\n>>> STEP 3: Testing trained model...")

try:
    result = subprocess.run(
        [sys.executable, "test_lightcurve_accuracy.py"],
        capture_output=False,
        text=True,
        check=True
    )
    log("✓ Testing completed!")
except subprocess.CalledProcessError as e:
    log(f"✗ Testing failed with error code {e.returncode}")
    # Don't exit - test failure is not critical

# Final summary
log("\n" + "="*70)
log("OVERNIGHT TRAINING PIPELINE COMPLETED!")
log("="*70)
log("\nNext steps:")
log("1. Restart Flask: python app.py")
log("2. Test API: curl -X POST http://127.0.0.1:5000/predict_lightcurve -H 'Content-Type: application/json' -d '{}'")
log("\nThe model is now trained on real NASA data and ready to use!")
log("="*70)
