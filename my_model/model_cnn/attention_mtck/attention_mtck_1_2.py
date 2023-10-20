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

# channelAttentionBlock
class CABLK(nn.Module):
    def __init__(self, in_channel, ratio=3):
        super(CABLK, self).__init__()
        self.maxPool = nn.AdaptiveMaxPool1d(1)
        self.minPool = nn.AdaptiveAvgPool1d(1)
        self.shared_mlp1 = nn.Conv1d(in_channel, in_channel // ratio, kernel_size=(1,), stride=(1, ))
        self.relu = nn.ReLU()
        self.shared_mlp2 = nn.Conv1d(in_channel // ratio, in_channel, kernel_size=(1,), stride=(1, ))
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X1 = self.maxPool(X)
        X2 = self.minPool(X)
        X1 = self.shared_mlp2(self.relu(self.shared_mlp1(X1)))
        X2 = self.shared_mlp2(self.relu(self.shared_mlp1(X2)))
        out = self.sigmoid(X1 + X2)
        return out * X

# spatialAttentionBlock
class SABLK(nn.Module):
    def __init__(self):
        super(SABLK, self).__init__()

        self.conv = nn.Conv1d(2, 1, kernel_size=(3, ), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X1 = torch.mean(X, dim=1, keepdim=True)
        X2, _ = torch.max(X, dim=1, keepdim=True)
        out = torch.cat([X1, X2], dim=1)
        out = self.sigmoid(self.conv(out))
        return X * out

b1 = nn.Sequential(
    nn.Conv1d(18, 64, kernel_size=(5,)),
    nn.BatchNorm1d(64), nn.ReLU(),
    # nn.MaxPool1d(kernel_size=(3,), stride=2, padding=1),
    # nn.Conv1d(64, 64, kernel_size=(1,)),
    # nn.BatchNorm1d(64), nn.ReLU(),
    nn.Conv1d(64, 192, kernel_size=(3,)),
    nn.BatchNorm1d(192), nn.ReLU(),
    # nn.MaxPool1d(kernel_size=(3,), stride=2, padding=1)
)

b2 = nn.Sequential(
    MTC(192, 64, (96, 128), (16, 32), 32),
    MTC(256, 128, (128, 192), (32, 96), 64),
    CABLK(480), SABLK(),
    # nn.MaxPool1d(kernel_size=(3,), stride=(2,), padding=1),
    MTC(480, 192, (96, 208), (16, 48), 64),
    MTC(512, 160, (112, 224), (24, 64), 64),
    CABLK(512), SABLK(),
    MTC(512, 128, (128, 256), (24, 64), 64),
    # MTC(512, 112, (144, 288), (32, 64), 64),
    # MTC(528, 256, (160, 320), (32, 128), 128),
    # nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
    # MTC(832, 256, (160, 320), (32, 128), 128),
    # MTC(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool1d(1),
    nn.Flatten()
)

net = nn.Sequential(b1, b2, nn.Linear(512, 8))