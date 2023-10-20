import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GatedGraphConv
import os.path as osp
import torch
from torch_geometric.data import Dataset, download_url

# 定义模型
class GatedGCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GatedGCN, self).__init__()
        self.conv1 = GatedGraphConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GatedGraphConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 加载数据集
# dataset = TUDataset(root='tmp/ZINC', name='ZINC_test')
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# 划分数据集为训练集和测试集
train_dataset = dataset[:500]
test_dataset = dataset[100:]
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 初始化模型并定义优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GatedGCN(hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 训练过程
def train(model, loader, optimizer):
    model.train()

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()


# 测试过程
def test(model, loader):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
        correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)


for epoch in range(1, 201):
    train(model, train_loader, optimizer)
    acc = test(model, test_loader)
    print(f'Epoch: {epoch}, Test Acc: {acc:.4f}')




class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]

    def download(self):
        # Download to `self.raw_dir`.
        path = download_url(url, self.raw_dir)
        ...

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = Data(...)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data

