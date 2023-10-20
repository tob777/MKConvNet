import torch
from torch import nn

class SEBLK(nn.Module):
    def __init__(self, input_Channel, ratio = 16):
        super(SEBLK, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.down = nn.Sequential(
            nn.Conv1d(input_Channel, input_Channel // ratio, kernel_size=(1,)),
            nn.ReLU()
        )
        self.up = nn.Sequential(
            nn.Conv1d(input_Channel // ratio, input_Channel, kernel_size=(1,)),
            nn.Sigmoid()
        )

    def forward(self, X):
        Y = self.avg_pool(X)
        Y = self.down(Y)
        Y = self.up(Y)

        return X * Y
