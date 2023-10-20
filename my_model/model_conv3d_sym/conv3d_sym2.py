import torch
from torch import nn

class My_Conv3d_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 5), stride=(1, 2, 2)), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(1, 1, 3), stride=(1, 1, 2)), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2)), nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1)), nn.ReLU(),
            # nn.Conv3d(128, 128, kernel_size=(3, 3, 6)), nn.BatchNorm3d(128), nn.ReLU(),
            # nn.Conv3d(128, 256, kernel_size=(3, 3, 6), stride=(2, 1, 1)), nn.BatchNorm3d(256), nn.ReLU(),
        )
        self.pool3d = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(128, 6)

    def forward(self, X):
        X = X.permute(0, 1, 2, 4, 3)
        X = self.conv3d(X)
        # X = X.squeeze()
        X = self.pool3d(X)
        X = self.fc(X)

        return X

My_Model = My_Conv3d_Model()

net = My_Model

X = torch.rand(64, 1, 3, 150, 18)
X = net(X)
print(X.shape)