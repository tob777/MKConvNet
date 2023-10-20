from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use1x1_blk=False, strides=1):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, num_channels, kernel_size=(3,), padding=1, stride=(strides,))
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=(3,), padding=1)
        if use1x1_blk:
            self.conv3 = nn.Conv1d(input_channels, num_channels, kernel_size=(1,), stride=(strides,))
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm1d(num_channels)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# blk = Residual(36, 36)
# X = torch.rand(2, 36, 150)
# Y = blk(X)
# print(Y.shape)
#
# blk = Residual(36, 72, use1x1_blk=True, strides=2)
# Y = blk(X)
# print(Y.shape)

b1 = nn.Sequential(
    nn.Conv1d(36, 64, kernel_size=(5,)),
    nn.BatchNorm1d(64), nn.ReLU(),
    nn.MaxPool1d(kernel_size=2)
)

# X = torch.rand(2, 36, 150)
# Y = b1(X)
# print(Y.shape)

def resnet_block(input_channels, num_channels, num_residual, first_block=False):
    blk = []
    for i in range(num_residual):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use1x1_blk=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))

    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, 9)
)

# X = torch.rand(2, 36, 150)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, 'output shape \t', X.shape)

