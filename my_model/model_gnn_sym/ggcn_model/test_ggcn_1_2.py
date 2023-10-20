import os
import pandas as pd
import torch.nn as nn
import torch_geometric.nn as gnn
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as f
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class test_GGCN(torch.nn.Module):
    def __init__(self, num_layers):
        super(test_GGCN, self).__init__()
        self.ggc1 = gnn.GatedGraphConv(128, num_layers=num_layers)
        self.ggc2 = gnn.GatedGraphConv(256, num_layers=num_layers)
        # self.pool1 = gnn.TopKPooling(32, ratio=0.8)
        self.fc = nn.Linear(256, 6)

    def forward(self, graph):
        X, edge, batch = graph.x, graph.edge_index, graph.batch.to(device)
        Y = f.relu(self.ggc1(X, edge))
        # Y, edge_index, _, batch, _, _ = self.pool1(Y, edge)
        Y = self.ggc2(Y, edge)
        Y = gnn.global_max_pool(Y, batch)
        Y = self.fc(Y)
        return Y

net = test_GGCN(32)

# test = []
# edge = torch.tensor([[i for i in range(0, 149)], [i for i in range(1, 150)]], dtype=torch.long)
# for i in range(50):
#     data = Data(x=torch.rand(150, 18), edge_index=edge.contiguous(), y=torch.tensor([1]))
#     test.append(data)
#
# x = DataLoader(test, batch_size=32)
# for X in x:
#     net(X)
#     print(X.shape)