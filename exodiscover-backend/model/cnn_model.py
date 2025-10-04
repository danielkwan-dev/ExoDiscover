import torch
import torch.nn as nn

class CNNTimeSeries(nn.Module):
    def __init__(self, input_length=100):
        super(CNNTimeSeries, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(16 * ((input_length - 2) // 2), 2)  # 2 classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
