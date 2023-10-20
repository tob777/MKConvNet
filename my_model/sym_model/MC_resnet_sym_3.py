import torch
from torch import nn
from torch.nn import functional as F

class resnet(nn.Module):
    def __init__(self, input_channels, num_channels, use1x1_blk=False, strides=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_channels, num_channels, kernel_size=(3,), padding=1, stride=(strides,)),
            nn.BatchNorm1d(num_channels), nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=(3,), padding=1),
            nn.BatchNorm1d(num_channels),
        )
        if use1x1_blk:
            self.conv1x1 = nn.Conv1d(input_channels, num_channels, kernel_size=(1,), stride=(strides,))
        else:
            self.conv1x1 = None

    def forward(self, X):
        Y = self.conv(X)
        if self.conv1x1:
            X = self.conv1x1(X)
        Y += X

        return F.relu(Y)

# class resnet(nn.Module):
#     def __init__(self, input_channels, num_channels, use1x1_blk=False, strides=1):
#         super().__init__()
#
#         self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=(3,), padding=1, stride=(strides,))
#         self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=(3,), padding=1)
#         if use1x1_blk:
#             self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=(1,), stride=(strides,))
#         else:
#             self.conv3 = None
#
#         self.bn1 = nn.BatchNorm1d(num_channels)
#         self.bn2 = nn.BatchNorm1d(num_channels)
#
#     def forward(self, X):
#         Y = F.relu(self.bn1(self.conv1(X)))
#         Y = self.bn2(self.conv2(Y))
#         if self.conv3:
#             X = self.conv3(X)
#         Y += X
#         return F.relu(Y)

def resnet_block(input_channels, num_channels, num_blocks, first_block=False):
    blk = []
    for i in range(num_blocks):
        if i == 0 and not first_block:
            blk.append(resnet(input_channels, num_channels, use1x1_blk=True, strides=2))
        else:
            blk.append(resnet(num_channels, num_channels))

    return blk

overall_cb1 = nn.Sequential(
    nn.Conv1d(18, 64, kernel_size=(5,)),
    nn.BatchNorm1d(64), nn.ReLU(),
    nn.MaxPool1d(kernel_size=2)
)

sub_cb1 = nn.Sequential(
    nn.Conv1d(9, 64, kernel_size=(5,)),
    nn.BatchNorm1d(64), nn.ReLU(),
    nn.MaxPool1d(kernel_size=2)
)

cnn_layers = nn.Sequential(
    *resnet_block(64, 64, 2, first_block=True),
    *resnet_block(64, 128, 2),
    *resnet_block(128, 256, 2),
    *resnet_block(256, 512, 2)
)

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
        self.overall_channel = overall_cnn
        self.sub_channel1 = sub_cnn
        self.sub_channel2 = sub_cnn

    def forward(self, X):
        X1 = torch.cat((X[:, 0:3], X[:, 6:9], X[:, 12:15]), dim=1)
        X2 = torch.cat((X[:, 3:6], X[:, 9:12], X[:, 15:18]), dim=1)
        X1 = self.sub_channel1(X1)
        X2 = self.sub_channel2(X2)
        X3 = self.overall_channel(X)

        return torch.cat((X1, X2, X3), dim=1)


net = nn.Sequential(
    Multi_cha_model(),
    nn.Linear(1536, 1024), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(512, 6),
)