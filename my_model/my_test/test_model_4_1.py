import torch
from torch import nn


class MTC(nn.Module):
    def __init__(self, input_channels, c1, c2, c3, c4):
        super().__init__()

        self.kt1 = nn.Sequential(
            nn.Conv1d(input_channels, c1, kernel_size=(1,)),
            nn.BatchNorm1d(c1), nn.ReLU()
        )

        self.kt2 = nn.Sequential(
            nn.Conv1d(input_channels, c2[0], kernel_size=(1,)),
            nn.BatchNorm1d(c2[0]), nn.ReLU(),
            nn.Conv1d(c2[0], c2[1], kernel_size=(3,), padding=1),
            nn.BatchNorm1d(c2[1]), nn.ReLU()
        )

        self.kt3 = nn.Sequential(
            nn.Conv1d(input_channels, c3[0], kernel_size=(1,)),
            nn.BatchNorm1d(c3[0]), nn.ReLU(),
            nn.Conv1d(c3[0], c3[1], kernel_size=(5,), padding=2),
            nn.BatchNorm1d(c3[1]), nn.ReLU()
        )

        self.kt4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=(3,), stride=(1,), padding=1),
            nn.Conv1d(input_channels, c4, kernel_size=(1,)),
            nn.BatchNorm1d(c4), nn.ReLU()
        )

    def forward(self, X):
        X1 = self.kt1(X)
        X2 = self.kt2(X)
        X3 = self.kt3(X)
        X4 = self.kt4(X)
        Y = torch.cat((X1, X2, X3, X4), dim=1)

        return Y


def data_reconstruction(X):
    X = torch.add(X[:, 0::2, :], X[:, 1::2, :])
    X = X[:, 0:72, :].permute(0, 2, 1)
    return X


class gru_block(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru1 = nn.GRU(input_size=18, hidden_size=64, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True)
        self.gru_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(18, 64, kernel_size=(5,)),
            nn.BatchNorm1d(64), nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=(3,), stride=(2,)),
            nn.BatchNorm1d(128), nn.ReLU()
        )

        self.CNNB = nn.Sequential(
            MTC(192, 64, (96, 128), (16, 32), 32),
            MTC(256, 128, (128, 192), (32, 96), 64),
            # nn.MaxPool1d(kernel_size=(3,), stride=(2,), padding=1),
            MTC(480, 192, (96, 208), (16, 48), 64),
            MTC(512, 160, (112, 224), (24, 64), 64),
            MTC(512, 128, (128, 256), (24, 64), 64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

    def forward(self, X):
        X1 = self.conv1(X)
        X1 = self.conv2(X1)
        X2 = X.permute(0, 2, 1)
        X2, h_n = self.gru1(X2, None)

        X2 = data_reconstruction(X2)
        X3, h_n = self.gru2(X2.permute(0, 2, 1), None)
        X3 = self.gru_fc(X3.permute(0, 2, 1))

        X1 = torch.cat((X1, X2), dim=1)
        X1 = self.CNNB(X1)

        out = torch.cat((X1, X3), dim=1)

        return out


net = nn.Sequential(gru_block(), nn.Linear(640, 1024), nn.Linear(1024, 6))

# X = torch.rand(2, 18, 150)
#
# Y = net(X)
# print("stop")


