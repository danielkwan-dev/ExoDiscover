import torch
from model.cnn_model import CNNTimeSeries
from model.lstm_model import LSTMTimeSeries

# choose model type here: 'cnn' or 'lstm'
MODEL_TYPE = 'cnn'

if MODEL_TYPE == 'cnn':
    model = CNNTimeSeries(input_length=100)
elif MODEL_TYPE == 'lstm':
    model = LSTMTimeSeries(input_size=1, hidden_size=32)
else:
    raise ValueError("MODEL_TYPE must be 'cnn' or 'lstm'")

model.eval()  # set to evaluation mode

def predict_lightcurve(curve_data):
    """
    curve_data: numpy array or list of shape (100,) for dummy test
    Returns dummy prediction for now
    """
    # convert to torch tensor
    import numpy as np
    if isinstance(curve_data, list):
        curve_data = np.array(curve_data)
    x = torch.tensor(curve_data, dtype=torch.float32).unsqueeze(0)  # batch dimension

    if MODEL_TYPE == 'cnn':
        x = x.unsqueeze(1)  # add channel dimension for CNN
    elif MODEL_TYPE == 'lstm':
        x = x.unsqueeze(-1)  # add feature dimension for LSTM

    # dummy forward pass
    with torch.no_grad():
        output = model(x)

    # dummy return
    return {"label": "Exoplanet", "confidence": 0.85}
