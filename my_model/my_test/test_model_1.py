import torch
from torch import nn

class my_gru(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x = nn.Sequential(
            nn.Conv1d(18, 32, kernel_size=(1,)), nn.ReLU(),
            # nn.Conv1d(32, 64, kernel_size=(1,)), nn.ReLU(),
        )
        self.gru1 = nn.GRU(input_size=32, hidden_size=64, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        # self.gru3 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.conv3x = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=(3,), stride=(2,)),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.gru3 = nn.GRU(input_size=256, hidden_size=386, num_layers=1, batch_first=True)
        self.gru4 = nn.GRU(input_size=386, hidden_size=512, num_layers=1, batch_first=True)
        self.conv5x = nn.Sequential(
            nn.Conv1d(512, 640, kernel_size=(5,), stride=(2,)),
            nn.BatchNorm1d(640), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc = nn.Linear(640, 6)

    def forward(self, X):
        X = self.conv1x(X)
        X = X.permute(0, 2, 1)
        X, h_n = self.gru1(X, None)
        X, h_n = self.gru2(X, None)
        X = X.permute(0, 2, 1)
        # X, h_n = self.gru3(X, None)
        X = self.conv3x(X)
        X = X.permute(0, 2, 1)
        X, h_n = self.gru3(X, None)
        X, h_n = self.gru4(X, None)
        X = X.permute(0, 2, 1)
        X = self.conv5x(X)

        out = self.fc(X)
        # out = self.fc(X[:, -1, :])

        return out

net = my_gru()