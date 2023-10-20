import torch_geometric.nn as gnn
import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class my_GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = gnn.GraphConv(900, 1024)
        self.conv2 = gnn.GraphConv(1024, 1024)
        self.conv3 = gnn.GraphConv(1024, 2048)
        self.conv4 = gnn.GraphConv(2048, 2048)

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 6)
        )

    def forward(self, data):
        X, edge_index, batch = data.x, data.edge_index, data.batch.to(device)

        X = F.relu(self.conv1(X, edge_index))
        X = F.relu(self.conv2(X, edge_index))
        X = F.relu(self.conv3(X, edge_index))
        X = F.relu(self.conv4(X, edge_index))

        X = gnn.global_max_pool(X, batch)
        X = self.fc(X)

        return X

net = my_GCN()