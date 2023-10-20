import torch
from torch import nn
import torch.functional as F

class DeepConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.ConvLayer = nn.Sequential(
            nn.Conv2d(1, 64, (5, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (5, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (5, 1)), nn.ReLU(),
            nn.Conv2d(64, 64, (5, 1)), nn.ReLU(),
        )

        self.lstm_layer1 = nn.LSTM(input_size=1152, hidden_size=128, num_layers=1, batch_first=True)
        self.drop = nn.Sequential(
            nn.Dropout(0.5)
        )
        self.lstm_layer2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, X):
        X = X.unsqueeze(dim=1)
        X = self.ConvLayer(X)
        X = X.permute(0, 2, 1, 3)
        X = X.reshape(-1, 134, 1152)
        X, (n_h, n_c) = self.lstm_layer1(X, None)
        X = self.drop(X)
        X, (n_h, n_c) = self.lstm_layer2(X, None)
        out = self.fc(X[:, -1, :])

        return out

net = DeepConvLSTM()

# X = torch.randn(64, 150, 18)
#
# net(X)