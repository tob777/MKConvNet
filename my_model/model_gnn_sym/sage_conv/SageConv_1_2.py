import torch_geometric.nn as gnn
import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class my_sageCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = gnn.SAGEConv(900, 1024)
        self.conv2 = gnn.SAGEConv(1024, 1024)
        self.conv3 = gnn.SAGEConv(1024, 2048)
        self.conv4 = gnn.SAGEConv(2048, 2048)
        self.conv5 = gnn.SAGEConv(2048, 1024)
        self.conv6 = gnn.SAGEConv(1024, 1024)
        self.conv7 = gnn.SAGEConv(1024, 512)
        self.conv8 = gnn.SAGEConv(512, 6)

    def forward(self, data):
        X, edge_index, batch = data.x, data.edge_index, data.batch.to(device)

        X = F.relu(self.conv1(X, edge_index))
        X = F.relu(self.conv2(X, edge_index))
        X = F.relu(self.conv3(X, edge_index))
        X = F.relu(self.conv4(X, edge_index))
        X = F.relu(self.conv5(X, edge_index))
        X = F.relu(self.conv6(X, edge_index))
        X = F.relu(self.conv7(X, edge_index))
        X = self.conv8(X, edge_index)

        X = gnn.global_mean_pool(X, batch)

        return X