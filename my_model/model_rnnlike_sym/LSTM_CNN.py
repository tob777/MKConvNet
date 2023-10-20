import torch
from torch import nn

class LSTM_CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm1 = nn.LSTM(input_size=18, hidden_size=32, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=32, hidden_size=32, num_layers=1, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 5), stride=(1, 2)), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(1, 3), stride=(1, 1)), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(128, 2)
        )

    def forward(self, X):
        X, (n_h, n_c) = self.lstm1(X, None)
        X, (n_h, n_c) = self.lstm2(X, None)
        X = X.unsqueeze(dim=1)
        X = self.conv(X)
        Y = self.fc(X)

        return Y

# X = torch.randn(64, 128, 18)

net = LSTM_CNN()
# net(X)