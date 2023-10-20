import torch
from torch import nn

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