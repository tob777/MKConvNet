import torch
from torch import nn

class My_New_Conv3d_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(18, 71, kernel_size=(5,), stride=(2,)),
            nn.BatchNorm1d(71), nn.ReLU(),
            nn.Conv1d(71, 71, kernel_size=(3,)),
            nn.BatchNorm1d(71), nn.ReLU(),
        )
        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), stride=(2, 2)),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1)),
            nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(5, 5, 5), stride=(2, 2, 2)),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(64), nn.ReLU(),
            # nn.Conv3d(64, 64, kernel_size=(3, 3, 3)),
            # nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(128), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 512), nn.ReLU(),
            nn.Linear(512, 6)

        )

    def forward(self, X):
        X = self.conv1d(X)
        X = X.unsqueeze(dim=1)
        X = self.conv2d(X)
        X = X.unsqueeze(dim=1)
        X = X.permute(0, 1, 4, 2, 3)
        X = self.conv3d(X)
        X = self.fc(X)

        return X

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = My_New_Conv3d_Model()
# X = torch.rand(64, 18, 150)
# X = net(X)