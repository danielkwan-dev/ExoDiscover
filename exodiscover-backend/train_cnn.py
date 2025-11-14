import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.cnn_model import CNNTimeSeries
from torch.utils.data import Dataset, DataLoader
from scipy import signal

# Simple dataset class
class LightCurveDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = labels

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # Resample to 100 points for model
        window = self.windows[idx]
        if len(window) != 100:
            window = signal.resample(window, 100)

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Training function
def train_model(model, train_loader, epochs=20, class_weights=None):
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Standard learning rate for deeper model

    model.train()
    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(epochs):
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

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%')

    print("=" * 60)
    return model

# Main training
if __name__ == "__main__":
    print("CNN Training Script for Exoplanet Detection")
    print("=" * 60)

    # Load training data
    print("Loading training data...")
    try:
        windows = np.load('training_data/windows.npy')
        labels = np.load('training_data/labels.npy')
        print(f"Loaded {len(windows)} windows")
        print(f"  - Confirmed (label=2): {sum(labels == 2)}")
        print(f"  - Candidate (label=1): {sum(labels == 1)}")
        print(f"  - False Positive (label=0): {sum(labels == 0)}")
    except FileNotFoundError:
        print("Training data not found!")
        print("Run 'python download_training_data.py' first to download data.")
        exit(1)

    # Create dataset and dataloader (smaller batch for better learning)
    dataset = LightCurveDataset(windows, labels)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model
    model = CNNTimeSeries(input_length=100)
    print(f"\nInitialized CNN model")
    print(f"  - Input length: 100 points")
    print(f"  - Output classes: 3 (False Positive / Candidate / Confirmed)")

    # Calculate class weights to handle imbalance
    num_false_positive = sum(labels == 0)
    num_candidate = sum(labels == 1)
    num_confirmed = sum(labels == 2)
    total = len(labels)

    weight_fp = total / (3 * num_false_positive) if num_false_positive > 0 else 1.0
    weight_cand = total / (3 * num_candidate) if num_candidate > 0 else 1.0
    weight_conf = total / (3 * num_confirmed) if num_confirmed > 0 else 1.0

    class_weights = torch.FloatTensor([weight_fp, weight_cand, weight_conf])
    print(f"\nCalculated class weights to handle imbalance:")
    print(f"  - False Positive (0) weight: {weight_fp:.2f}")
    print(f"  - Candidate (1) weight: {weight_cand:.2f}")
    print(f"  - Confirmed (2) weight: {weight_conf:.2f}")

    # Train - with improved data and model, 30 epochs should be enough
    model = train_model(model, train_loader, epochs=30, class_weights=class_weights)

    # Save trained model
    torch.save(model.state_dict(), 'model/trained_cnn.pth')
    print(f"\nModel saved to model/trained_cnn.pth")

    # Test on a few samples
    print("\nTesting on random samples...")
    model.eval()
    label_map = {0: "False Positive", 1: "Candidate", 2: "Confirmed"}
    with torch.no_grad():
        for i in range(5):
            idx = np.random.randint(0, len(windows))
            x, y = dataset[idx]
            x = x.unsqueeze(0)  # Add batch dimension
            output = model(x)
            _, predicted = torch.max(output.data, 1)
            prob = torch.softmax(output, dim=1)[0]

            actual = label_map[y.item()]
            pred = label_map[predicted.item()]
            confidence = prob[predicted].item()

            print(f"  Sample {i+1}: Actual={actual:15s} | Predicted={pred:15s} | Confidence={confidence:.2%}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("\nTo use the trained model in Flask:")
    print("1. Update predict.py to load the weights:")
    print("   model.load_state_dict(torch.load('model/trained_cnn.pth'))")
    print("2. Restart Flask: python app.py")
    print("=" * 60)
