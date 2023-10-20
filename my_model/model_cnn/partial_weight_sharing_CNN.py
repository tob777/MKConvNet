import torch
from torch import nn
import torch.nn.functional as F

class PWS_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convp1 = nn.Conv2d(1, 16, kernel_size=(3, 3))
        self.convp2 = nn.Conv2d(1, 16, kernel_size=(3, 3))
        self.convp3 = nn.Conv2d(1, 16, kernel_size=(3, 3))
        self.padding1 = nn.ZeroPad2d((0, 0, 0, 2))

        self.pooling1 = nn.MaxPool2d((2, 3))
        self.pooling2 = nn.MaxPool2d((2, 3))
        self.pooling3 = nn.MaxPool2d((2, 3))

        self.padding2 = nn.ZeroPad2d((0, 0, 6, 0))
        self.padding3 = nn.ZeroPad2d((0, 0, 0, 6))

        self.conv = nn.Conv2d(48, 64, kernel_size=(3, 5))
        self.pooling4 = nn.MaxPool2d((2, 3))

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 2)
        )

    def forward(self, X):
        X1 = torch.cat((X[:, 0:3], X[:, 6:9], X[:, 12:15]), dim=1)
        X1 = X1.unsqueeze(dim=1)
        X2 = torch.cat((X[:, 3:6], X[:, 9:12], X[:, 15:18]), dim=1)
        X2 = X2.unsqueeze(dim=1)
        X3 = torch.cat(((self.padding1(X1)), X2), dim=2)

        X1 = F.relu(self.convp1(X1))
        X2 = F.relu(self.convp2(X2))
        X3 = F.relu(self.convp3(X3))

        X1 = self.pooling1(X1)
        X1 = self.padding2(X1)

        X2 = self.pooling2(X2)
        X2 = self.padding3(X2)

        X3 = self.pooling3(X3)

        X = torch.cat((X1, X3, X2), dim=1)
        X = F.relu(self.conv(X))
        X = self.pooling4(X)
        Y = self.fc(X)

        return Y

net = PWS_CNN()

# X = torch.rand(size=(1, 18, 60))
# net(X)
