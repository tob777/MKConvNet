from torch import nn
import torch

class Muti_Channels(nn.Module):
    def __init__(self):
        super().__init__()
        self.add_channel = nn.Sequential(
            nn.Conv3d(1, 512, kernel_size=(1, 1, 1)), nn.ReLU()
        )
        self.c1 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=(1, 1, 1)), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten()
        )
        self.c2 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=(1, 1, 1)), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 1, 1)), nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 1)), nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1, 1, 3)), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten()
        )
        self.c3 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=(1, 1, 1)), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(5, 1, 1)), nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1, 3, 1)), nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=(1, 1, 3)), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten()
        )
        self.c4 = nn.Sequential(
            nn.Conv3d(512, 64, kernel_size=(1, 1, 1)), nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3)), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten()
        )
        self.c5 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3)),
            nn.Conv3d(512, 64, kernel_size=(1, 1, 1)), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten()
        )
        self.fc = nn.Linear(512, 6)

    def forward(self, X):
        X = self.add_channel(X)
        X1 = self.c1(X)
        X2 = self.c2(X)
        X3 = self.c3(X)
        X4 = self.c4(X)
        X5 = self.c5(X)

        Y = torch.cat((X1, X2, X3, X4, X5), dim=1)
        Y = self.fc(Y)

        return Y

My_Model = Muti_Channels()

net = My_Model

# X = torch.rand(64, 1, 150, 3, 6)
# X = net(X)
# print(X.shape)