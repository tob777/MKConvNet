from torch import nn

class my_cnn_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(18, 64, kernel_size=(5,), stride=(2,)), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=(3,), stride=(2,)), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=(3,)), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=(3,), stride=(2,)), nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=(3,), stride=(2,)), nn.ReLU(),
            nn.Flatten(),
        )
        self.Linear = nn.Sequential(
            nn.Linear(3584, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
        )
        self.fc = nn.Linear(1024, 6)

    def forward(self, X):
        X = self.fc(self.Linear(self.conv1d(X)))
        return X

net = my_cnn_model()