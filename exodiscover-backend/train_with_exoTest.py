"""
Train CNN using exoTest.csv dataset
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import signal
from model.cnn_model import CNNTimeSeries

print("Loading exoTest.csv...")
df = pd.read_csv('test_data/exoTest.csv')

# Extract labels
labels = []
for label in df['LABEL']:
    # Convert: 2 (exoplanet) -> 1, 1 (non-exoplanet) -> 0
    labels.append(1 if label == 2 else 0)

# Extract flux data
flux_cols = [col for col in df.columns if col.startswith('FLUX')]
flux_data = df[flux_cols].values

print(f"Loaded {len(df)} samples with {len(flux_cols)} flux points each")
print(f"  - Exoplanets (label=1): {sum(labels)}")
print(f"  - Non-exoplanets (label=0): {len(labels) - sum(labels)}")

# Resample all flux to 100 points for CNN
print("\nResampling flux data to 100 points...")
resampled_flux = []
for i in range(len(flux_data)):
    flux_100 = signal.resample(flux_data[i], 100)
    resampled_flux.append(flux_100)

resampled_flux = np.array(resampled_flux)

# Dataset class
class LightCurveDataset(Dataset):
    def __init__(self, flux_data, labels):
        self.flux_data = flux_data
        self.labels = labels

    def __len__(self):
        return len(self.flux_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.flux_data[idx], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Split into train/test
split_idx = int(0.8 * len(resampled_flux))
train_flux = resampled_flux[:split_idx]
train_labels = labels[:split_idx]
test_flux = resampled_flux[split_idx:]
test_labels = labels[split_idx:]

print(f"\nTrain set: {len(train_flux)} samples")
print(f"Test set: {len(test_flux)} samples")

# Create dataloaders
train_dataset = LightCurveDataset(train_flux, train_labels)
test_dataset = LightCurveDataset(test_flux, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model with class weights to handle imbalance
model = CNNTimeSeries(input_length=100)

# Calculate class weights (inverse of frequency)
num_exo = sum(train_labels)
num_non_exo = len(train_labels) - num_exo
weight_exo = len(train_labels) / (2 * num_exo) if num_exo > 0 else 1.0
weight_non_exo = len(train_labels) / (2 * num_non_exo) if num_non_exo > 0 else 1.0
class_weights = torch.FloatTensor([weight_non_exo, weight_exo])

print(f"\nClass weights: Non-exo={weight_non_exo:.2f}, Exo={weight_exo:.2f}")

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
print("\nTraining CNN...")
print("=" * 60)

for epoch in range(20):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    train_acc = 100 * correct / total
    avg_loss = total_loss / len(train_loader)

    # Test accuracy
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()

    test_acc = 100 * test_correct / test_total

    print(f"Epoch {epoch+1:2d}/20 | Loss: {avg_loss:.4f} | "
          f"Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

print("=" * 60)

# Save model
torch.save(model.state_dict(), 'model/trained_cnn.pth')
print("\nModel saved to model/trained_cnn.pth")
print("Restart Flask to use the updated model!")
