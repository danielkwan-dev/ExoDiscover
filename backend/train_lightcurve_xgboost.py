"""
Train XGBoost model on light curve FEATURES instead of raw data
This should work much better than CNN on raw patterns
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

print("="*70)
print("TRAINING XGBOOST ON LIGHT CURVE FEATURES")
print("="*70)

# Load features
df = pd.read_csv('training_data/lightcurve_features.csv')

print(f"\nDataset: {len(df)} samples with {len(df.columns)-1} features")
print(f"  Confirmed (2): {sum(df['label']==2)}")
print(f"  Candidate (1): {sum(df['label']==1)}")
print(f"  False Positive (0): {sum(df['label']==0)}")

# Prepare data
X = df.drop('label', axis=1)
y = df['label']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train XGBoost
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("\n" + "="*70)
print("TRAINING RESULTS")
print("="*70)
print(f"Training accuracy: {train_accuracy:.2%}")
print(f"Test accuracy: {test_accuracy:.2%}")

print("\nTest Set Classification Report:")
label_names = ['False Positive', 'Candidate', 'Confirmed']
print(classification_report(y_test, y_pred_test, target_names=label_names))

print("\nConfusion Matrix (Test Set):")
cm = confusion_matrix(y_test, y_pred_test)
print("                 Predicted")
print("               FP   Cand  Conf")
print(f"Actual FP     {cm[0][0]:3d}   {cm[0][1]:3d}   {cm[0][2]:3d}")
print(f"       Cand   {cm[1][0]:3d}   {cm[1][1]:3d}   {cm[1][2]:3d}")
print(f"       Conf   {cm[2][0]:3d}   {cm[2][1]:3d}   {cm[2][2]:3d}")

# Feature importance
print("\nTop 10 Most Important Features:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

# Save model
joblib.dump(model, 'model/lightcurve_xgb_model.pkl')
print(f"\nModel saved to model/lightcurve_xgb_model.pkl")
print("="*70)
