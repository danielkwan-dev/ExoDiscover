import pandas as pd
import numpy as np

# Load the merged test dataset
df = pd.read_csv("merged_exoplanets.csv")

# -----------------------------
# Dummy prediction function
# -----------------------------
def dummy_predict(row):
    """
    Simulate planet disposition prediction:
    - 2 (Confirmed) if planet_radius > 1.5
    - 1 (Candidate) if 1.0 < planet_radius <= 1.5
    - 0 (False Positive) if planet_radius <= 1.0
    """
    if row['planet_radius'] > 1.5:
        return 2
    elif row['planet_radius'] > 1.0:
        return 1
    else:
        return 0

# Apply dummy prediction
df['predicted_disposition'] = df.apply(dummy_predict, axis=1)

# Map numeric labels back to strings for readability
label_map = {2: 'Confirmed', 1: 'Candidate', 0: 'False Positive'}
df['predicted_label'] = df['predicted_disposition'].map(label_map)

# -----------------------------
# Sample a few rows to print
# -----------------------------
sample_df = df.sample(n=10, random_state=42)  # pick 10 random rows
print("✅ Predictions for 10 random rows from test dataset:\n")
print(sample_df[['planet_name', 'planet_radius', 'predicted_label', 'source']])

# Optionally save predictions
df.to_csv("test_predictions.csv", index=False)
print("\n💾 Predictions saved to 'test_predictions.csv'")
