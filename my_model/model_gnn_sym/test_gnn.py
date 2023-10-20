from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

x = DataLoader(dataset, batch_size=1)

for data in x:
    print(data.num_nodes)