import torch
from torch import nn

class cnn(nn.Module):
    def __init__(self, input_channels, num_channels, strides=2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, num_channels, kernel_size=(3,), stride=(strides,)), nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=(3,)), nn.ReLU(),
        )

    def forward(self, X):
        X = self.conv(X)
        return X

def cnn_block(input_channels, num_channels, num_blocks):
    blk = []
    for i in range(num_blocks):
        if i == 0:
            blk.append(cnn(input_channels, num_channels))
        else:
            blk.append(cnn(num_channels, num_channels, strides=1))

    return blk

overall_cb1 = nn.Sequential(
    nn.Conv1d(18, 64, kernel_size=(5,)), nn.ReLU(),
    nn.MaxPool1d(kernel_size=2)
)

sub_cb1 = nn.Sequential(
    nn.Conv1d(9, 64, kernel_size=(5,)), nn.ReLU(),
    nn.MaxPool1d(kernel_size=2)
)

cnn_layers = nn.Sequential(*cnn_block(64, 128, 2), *cnn_block(128, 256, 2))
overall_cnn = nn.Sequential(
    overall_cb1, cnn_layers,
    nn.AdaptiveAvgPool1d(1), nn.Flatten()
)
sub_cnn = nn.Sequential(
    sub_cb1, cnn_layers,
    nn.AdaptiveAvgPool1d(1), nn.Flatten()
)

class Multi_cha_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_1 = overall_cnn
        self.channel_2 = overall_cnn

    def forward(self, X):
        X1 = torch.cat((X[:, 0:3], X[:, 6:9], X[:, 12:15], X[:, 3:6], X[:, 9: 12], X[:, 15:18]), dim=1)
        X1 = self.channel_1(X1)
        X2 = self.channel_2(X)

        return torch.cat((X1, X2), dim=1)


net = nn.Sequential(
    Multi_cha_model(),
    nn.Linear(512, 6) #, nn.ReLU(), nn.Dropout(0.5),
    # nn.Linear(512, 6)
)