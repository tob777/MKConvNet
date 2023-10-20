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

class b1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(18, 64, kernel_size=(5,)),
            nn.BatchNorm1d(64), nn.ReLU(),
            # nn.MaxPool1d(kernel_size=(3,)),
            # nn.Conv1d(64, 64, kernel_size=(1,)), nn.ReLU(),
        )
        self.grublk = nn.GRU(input_size=64, hidden_size=192, num_layers=1, batch_first=True)
        # self.pool = nn.Sequential(
        #     nn.BatchNorm1d(192), nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=(3,), stride=2, padding=1)
        # )

    def forward(self, X):
        X = self.conv(X)
        X = X.permute(0, 2, 1)
        X, h_n = self.grublk(X)
        # X = X[:, 6:38]
        X = X.permute(0, 2, 1)
        # X = self.pool(X)

        return X

b2 = nn.Sequential(
    MTC(192, 64, (96, 128), (16, 32), 32),
    MTC(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool1d(kernel_size=(3,), stride=(2,), padding=1),
    MTC(480, 192, (96, 208), (16, 48), 64),
    MTC(512, 160, (112, 224), (24, 64), 64),
    MTC(512, 128, (128, 256), (24, 64), 64),
    # MTC(512, 112, (144, 288), (32, 64), 64),
    # MTC(528, 256, (160, 320), (32, 128), 128),
    # nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
    # MTC(832, 256, (160, 320), (32, 128), 128),
    # MTC(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten()
)

class my_gru(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = b1()
        self.mtck = b2
        # self.gru3 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(512, 6)

    def forward(self, X):
        X = X.permute(0, 2, 1)
        X = self.gru(X)
        X = self.mtck(X)
        # X, h_n = self.gru3(X, None)
        out = self.fc(X)

        return out

net = my_gru()

# X = torch.rand(64, 150, 18)
# X = net(X)