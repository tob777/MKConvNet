import torch
from torch import nn
from d2l import torch as d2l

class conv_sub_blk(nn.Module):
    def __init__(self, input_channels, num_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, num_channels, kernel_size=(5,), stride=(2,)), nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=(3,)), nn.ReLU(),
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=(3,), stride=(2,)), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )
    def forward(self, X):
        X = self.conv1(X)
        return X

class conv_overall_blk(nn.Module):
    def __init__(self, input_channels, num_channels):
        super().__init__()
        self.num_channels1 = num_channels * 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, num_channels, kernel_size=(5,), stride=(2,)), nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=(3,)), nn.ReLU(),
            nn.Conv1d(num_channels, num_channels * 2, kernel_size=(3,), stride=(2,)), nn.ReLU(),
            nn.Conv1d(self.num_channels1, self.num_channels1 * 2, kernel_size=(3,), stride=(2,)), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten()
        )

    def forward(self, X):
        X = self.conv1(X)
        return X

class My_cnn_model(nn.Module):
    def __init__(self, input_channels1, input_channels2, num_channels1, num_channels2):
        super().__init__()
        self.overall1 = conv_overall_blk(input_channels1, num_channels1)
        self.overall2 = conv_overall_blk(input_channels1, num_channels1)
        self.sub1 = conv_sub_blk(input_channels2, num_channels2)
        self.sub2 = conv_sub_blk(input_channels2, num_channels2)
        self.sub3 = conv_sub_blk(input_channels2, num_channels2)
        self.sub4 = conv_sub_blk(input_channels2, num_channels2)
        self.sub5 = conv_sub_blk(input_channels2, num_channels2)
        self.sub6 = conv_sub_blk(input_channels2, num_channels2)
    def forward(self, X):

        Y1 = torch.cat((X[:, 0:3], X[:, 30:33], X[:, 6:9], X[:, 24:27], X[:, 12:15], X[:, 18:21],
                       X[:, 3:6], X[:, 33:36], X[:, 9:12], X[:, 27:30], X[:, 15:18], X[:, 21:24]), dim=1)
        X_sub1 = self.sub1(Y1[:, 0:18])
        X_sub2 = self.sub2(Y1[:, 18:36])
        X1 = self.overall1(Y1)

        # x_sub3 = self.sub3
        # X1 = self.overall1(X)

        Y2 = torch.cat((X[:, 0:3], X[:, 6:9], X[:, 12:15], X[:, 18:21], X[:, 24:27], X[:, 30:33],
                       X[:, 3:6], X[:, 9:12], X[:, 15:18], X[:, 21:24], X[:, 27:30], X[:, 33:36]), dim=1)
        X_sub3 = self.sub3(Y2[:, 0:18])
        X_sub4 = self.sub4(Y2[:, 18:36])
        X2 = self.overall2(Y2)

        X_sub5 = self.sub5(torch.cat((Y2[:, 0:9], Y2[:, 18:27]), dim=1))
        X_sub6 = self.sub6(torch.cat((Y2[:, 9:18], Y2[:, 27:36]), dim=1))

        Y = torch.cat((X1, X2, X_sub1, X_sub2, X_sub3, X_sub4, X_sub5, X_sub6), dim=1)
        return Y

input_channels1, input_channels2, num_channels1, num_channels2 = 36, 18, 64, 64
net = nn.Sequential(
    My_cnn_model(input_channels1, input_channels2, num_channels1, num_channels2),
    nn.Linear(1280, 1024), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(1024, 9)
)



# x = torch.rand(1, 36, 150)
# y = net(x)
# print(y.shape)

