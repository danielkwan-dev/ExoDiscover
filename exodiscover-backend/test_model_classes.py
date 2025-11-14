import numpy as np
import torch
from model.cnn_model import CNNTimeSeries
from scipy import signal

model = CNNTimeSeries(input_length=100)
model.load_state_dict(torch.load('model/trained_cnn.pth'))
model.eval()

windows = np.load('training_data/windows.npy')
labels = np.load('training_data/labels.npy')

predictions = []
for i in range(30):
    window = windows[i]
    if len(window) != 100:
        window = signal.resample(window, 100)

    with torch.no_grad():
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        output = model(x)
        pred = torch.argmax(output, dim=1).item()
        predictions.append(pred)

print('Predictions:', predictions)
print('True labels:', labels[:30].tolist())
print('Distribution - False Positive:', predictions.count(0), '| Candidate:', predictions.count(1), '| Confirmed:', predictions.count(2))
