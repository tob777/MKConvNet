import torch
from torch import nn

class my_gru(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru1 = nn.GRU(input_size=18, hidden_size=64, num_layers=1, batch_first=True)
        self.gru2 = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        # self.gru3 = nn.GRU(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, 6)

    def forward(self, X):
        X, h_n = self.gru1(X, None)
        X, h_n = self.gru2(X, None)
        # X, h_n = self.gru3(X, None)
        out = self.fc(X[:, -1, :])

        return out

net = my_gru()

# X = torch.rand(2, 150, 18)
# net(X)