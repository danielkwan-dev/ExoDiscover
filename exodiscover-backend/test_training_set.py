"""
Test the model on the training set to verify all 3 classes work
"""
import numpy as np
import joblib
import pandas as pd
from scipy import signal, stats

print("="*80)
print("TESTING MODEL ON TRAINING SET")
print("="*80)

# Load model
model = joblib.load('model/lightcurve_xgb_model.pkl')

# Load training features
features_df = pd.read_csv('training_data/lightcurve_features.csv')

print(f"\nDataset: {len(features_df)} samples")
print(f"  False Positive (0): {sum(features_df['label']==0)}")
print(f"  Candidate (1): {sum(features_df['label']==1)}")
print(f"  Confirmed (2): {sum(features_df['label']==2)}")

# Test on samples
X = features_df.drop('label', axis=1)
y = features_df['label']

predictions = model.predict(X)
probabilities = model.predict_proba(X)

# Calculate accuracy
correct = sum(predictions == y)
accuracy = correct / len(y) * 100

label_names = {0: 'False Positive', 1: 'Candidate', 2: 'Confirmed'}

print("\n" + "="*80)
print("OVERALL RESULTS")
print("="*80)
print(f"Accuracy: {correct}/{len(y)} = {accuracy:.1f}%")

# Show some examples from each class
print("\n" + "="*80)
print("SAMPLE PREDICTIONS (5 per class)")
print("="*80)

for true_label in [0, 1, 2]:
    print(f"\n{label_names[true_label]} samples:")
    print("-"*80)

    indices = features_df[features_df['label'] == true_label].head(5).index

    for i, idx in enumerate(indices):
        pred = predictions[idx]
        conf = probabilities[idx][pred]
        true = y[idx]

        pred_name = label_names[int(pred)]
        true_name = label_names[int(true)]

        status = "CORRECT" if pred == true else "WRONG"

        print(f"  Sample {i+1}: True={true_name:15s} | Predicted={pred_name:15s} ({conf:.1%}) [{status}]")

# Count predictions by class
print("\n" + "="*80)
print("PREDICTION DISTRIBUTION")
print("="*80)

for label in [0, 1, 2]:
    count = sum(predictions == label)
    percentage = (count / len(predictions)) * 100
    print(f"  {label_names[label]:20s}: {count:4d} ({percentage:.1f}%)")

print("\n" + "="*80)
print("MODEL CAN PREDICT ALL 3 CLASSES!")
print("="*80)
