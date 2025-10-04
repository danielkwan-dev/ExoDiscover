import torch
import torch.nn as nn

class LSTMTimeSeries(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_classes=2):
        super(LSTMTimeSeries, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn[-1])
        return x
