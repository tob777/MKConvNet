import torch
from torch import nn
from torch.nn import functional as f

class MTC(nn.Module):
    def __init__(self, input_channels, c1, c2, c3, c4):
        super().__init__()

        self.kt1 = nn.Sequential(
            nn.Conv1d(input_channels, c1, kernel_size=(1,)),
            nn.BatchNorm1d(c1), nn.ReLU()
        )
        self.kt2 = nn.Sequential(
            nn.Conv1d(input_channels, c2[0], kernel_size=(1,)), nn.ReLU(),
            nn.Conv1d(c2[0], c2[1], kernel_size=(3,), padding=1),
            nn.BatchNorm1d(c2[1]), nn.ReLU()
        )
        self.kt3 = nn.Sequential(
            nn.Conv1d(input_channels, c3[0], kernel_size=(1,)), nn.ReLU(),
            nn.Conv1d(c3[0], c3[1], kernel_size=(5, ), padding=2),
            nn.BatchNorm1d(c3[1]), nn.ReLU()
        )
        self.kt4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=(3,), stride=(1,), padding=1),
            nn.Conv1d(input_channels, c4, kernel_size=(1,)),
            nn.BatchNorm1d(c4), nn.ReLU()
        )
        # self.resconv = nn.Conv1d(input_channels, c1 + c2[1] + c3[1] + c4, kernel_size=(1,))

    def forward(self, X):
        X1 = self.kt1(X)
        X2 = self.kt2(X)
        X3 = self.kt3(X)
        X4 = self.kt4(X)
        Y = torch.cat((X1, X2, X3, X4), dim=1)
        # X = self.resconv(X)
        # Y += X
        return Y

# b1 = nn.Sequential(
#     # nn.Conv1d(18, 64, kernel_size=(5,)),
#     # nn.BatchNorm1d(64), nn.ReLU(),
#     # nn.MaxPool1d(kernel_size=(3,), stride=2, padding=1),
#     # nn.Conv1d(64, 64, kernel_size=(1,)), nn.ReLU(),
#     nn.Conv1d(64, 192, kernel_size=(3,)),
#     nn.BatchNorm1d(192), nn.ReLU(),
#     # nn.MaxPool1d(kernel_size=(3,), stride=2, padding=1)
# )

b2 = nn.Sequential(
    MTC(96, 32, (48, 64), (8, 16), 16),
    MTC(128, 64, (64, 96), (16, 48), 32),
    # nn.MaxPool1d(kernel_size=(3,), stride=(2,), padding=1),
    MTC(240, 96, (48, 104), (8, 24), 32),
    MTC(256, 80, (56, 112), (12, 32), 32),
    MTC(256, 64, (64, 128), (12, 32), 32),
    # MTC(512, 112, (144, 288), (32, 64), 64),
    # MTC(528, 256, (160, 320), (32, 128), 128),
    # nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
    # MTC(832, 256, (160, 320), (32, 128), 128),
    # MTC(832, 384, (192, 384), (48, 128), 128),
    # nn.AdaptiveAvgPool1d(1),
    # nn.Flatten()
)

def data_reconstruction(X):
    X = torch.add(X[:, 0::2, :], X[:, 1::2, :])
    return X

class my_gru(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=18, hidden_size=96, num_layers=1, batch_first=True)
        self.mtck = nn.Sequential(b2)
        self.gru2 = nn.GRU(input_size=256, hidden_size=512, num_layers=1, batch_first=True)
        # self.gru3 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.output_handle = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(512, 6)

    def forward(self, X):
        # X = data_reconstruction(X)

        X, h_n = self.gru1(X, None)
        # X, h_n = self.gru2(X, None)

        X = X.permute(0, 2, 1)
        X = self.mtck(X)
        # X, h_n = self.gru3(X, None)
        X = X.permute(0, 2, 1)
        X, h_n = self.gru2(X, None)
        X = X.permute(0, 2, 1)
        X = self.output_handle(X)

        out = self.fc(X)

        return out

net = my_gru()

# X = torch.rand(64, 150, 18)
# X = net(X)