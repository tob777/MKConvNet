import torch
from torch import nn

class my_lstm(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=18, hidden_size=64, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=4, batch_first=True)
        self.fc = nn.Linear(128, 6)

    def forward(self, X):
        X, (n_h, n_c) = self.lstm1(X, None)
        X, (n_h, n_c) = self.lstm2(X, None)
        out = self.fc(X[:, -1, :])

        return out

net = my_lstm()

# X = torch.rand(50, 150, 18)
#
# net(X)