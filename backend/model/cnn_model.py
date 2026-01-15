import torch
import torch.nn as nn

class CNNTimeSeries(nn.Module):
    """
    Improved CNN for detecting transit dips in exoplanet light curves.
    Deeper architecture with multiple conv layers to better learn patterns.
    """
    def __init__(self, input_length=100):
        super(CNNTimeSeries, self).__init__()

        # First conv block - detect local patterns
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)

        # Second conv block - combine local patterns
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)

        # Third conv block - higher level features
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)

        # Calculate flattened size after 3 pooling operations
        # input_length -> pool -> pool -> pool
        final_length = input_length // 2 // 2 // 2
        self.fc1 = nn.Linear(128 * final_length, 256)
        self.fc2 = nn.Linear(256, 3)  # 3 classes: False Positive, Candidate, Confirmed

    def forward(self, x):
        # Conv block 1
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))

        # Conv block 2
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))

        # Conv block 3
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)

        return x
