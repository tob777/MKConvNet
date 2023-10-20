import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l
import draw_conf_matrix as dcm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data1 = pd.read_csv('D:/1.data/11_train_sets/data_unemg_all.csv')
# data2 = pd.read_csv('D:/2.data/11_train_sets/unemg_all_1.csv')
# data3 = pd.read_csv('D:/2.data/11_train_sets/unemg_all_2.csv')
data_all = data1 #pd.concat((data1, data2, data3))

features = data_all.iloc[:, 0: -1]
col = features.columns
features = features[col].apply(lambda x: (x - x.mean()) / (x.std()))

n_train = (features.shape[0] // 5) * 3

X_train = torch.tensor(features[:n_train].values, dtype=torch.float32).to(device)
y_train = torch.tensor(data_all[:n_train].label.values.reshape(-1, 1), dtype=torch.float32).to(device)
X_test = torch.tensor(features[n_train:].values, dtype=torch.float32).to(device)
y_test = torch.tensor(data_all[n_train:].label.values.reshape(-1, 1), dtype=torch.float32).to(device)

input1, input2 = 12, 18
hidden1, hidden2, hidden3, hidden4, output1 = 80, 80, 40, 40, 16
hidden5, hidden6, output = (output1 * 6), 36, 9
loss = nn.CrossEntropyLoss()

class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Linear(input1, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, hidden3), nn.ReLU(),
            nn.Linear(hidden3, hidden4), nn.ReLU(),
            nn.Linear(hidden4, output1), nn.ReLU()
        )
        self.net2 = nn.Sequential(
            nn.Linear(input2, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, hidden3), nn.ReLU(),
            nn.Linear(hidden3, hidden4), nn.ReLU(),
            nn.Linear(hidden4, output1), nn.ReLU(),
        )
        self.net3 = nn.Sequential(
            nn.Linear(input2, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, hidden3), nn.ReLU(),
            nn.Linear(hidden3, hidden4), nn.ReLU(),
            nn.Linear(hidden4, output1), nn.ReLU(),
        )
        self.net4 = nn.Sequential(
            nn.Linear(input2, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, hidden3), nn.ReLU(),
            nn.Linear(hidden3, hidden4), nn.ReLU(),
            nn.Linear(hidden4, output1), nn.ReLU(),
        )
        self.net5 = nn.Sequential(
            nn.Linear(input2, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, hidden3), nn.ReLU(),
            nn.Linear(hidden3, hidden4), nn.ReLU(),
            nn.Linear(hidden4, output1), nn.ReLU(),
        )
        self.net6 = nn.Sequential(
            nn.Linear(input1, hidden1), nn.ReLU(),
            nn.Linear(hidden1, hidden2), nn.ReLU(),
            nn.Linear(hidden2, hidden3), nn.ReLU(),
            nn.Linear(hidden3, hidden4), nn.ReLU(),
            nn.Linear(hidden4, output1), nn.ReLU(),
        )


    def forward(self, X):
        x1, x2, x3, x4, x5, x6 = X[:, 0: 12], X[:, 0: 18], X[:, 6: 24], X[:, 12: 30], X[:, 18: 36], X[:, 24: 36]
        X1 = self.net1(x1)
        X2 = self.net2(x2)
        X3 = self.net3(x3)
        X4 = self.net4(x4)
        X5 = self.net5(x5)
        X6 = self.net6(x6)
        return torch.cat((X1, X2, X3, X4, X5, X6), dim=1)

net = nn.Sequential(
    MyMLP(),
    nn.Linear(hidden5, hidden6), nn.ReLU(),
    nn.Linear(hidden6, output)
    )

def get_k_fold_data(K, i, X, y):
    assert K > 1
    fold_size = X.shape[0] // K

    start = i * fold_size
    end = (i + 1) * fold_size
    if i == 0:
        X_valid, y_valid = X[:end, :], y[:end]
        X_train, y_train = X[end:, :], y[end:]
    elif i == K - 1:
        X_valid, y_valid = X[start:, :], y[start:]
        X_train, y_train = X[:start, :], y[:start]
    else:
        X_valid, y_valid = X[start:end, :], y[start:end]
        X_train, y_train = torch.cat([X[:start, :], X[end:]], 0), torch.cat([y[:start], y[end:]], 0)

    return X_train, y_train, X_valid, y_valid

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)

    y = y.reshape(len(y))
    cmp = d2l.astype(y_hat, y.dtype) == y

    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))

def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset.

    Defined in :numref:`sec_softmax_scratch`"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    # conf_matrix, test_acc = torch.zeros(9, 9), []
    with torch.no_grad():
        for X, y in data_iter:
            # y = y.squeeze()
            #
            # # 将变量转为gpu
            X.to(device)
            y.to(device)

            y_hat = net(X).to(device)
            metric.add(accuracy(y_hat, y), d2l.size(y))

            # 记录混淆矩阵参数
    #         conf_matrix = dcm.confusion_matrix(y_hat, y, conf_matrix)
    #         conf_matrix = conf_matrix.cpu()
    #
    # labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # dcm.plot_confusion_matrix(conf_matrix, 9, labels)

    # with torch.no_grad():
    #     for X, y in data_iter:
    #         metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]

def load_array(data_arrays, batch_size, is_train=False):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def train(net, X_train, y_train, X_valid, y_valid, weight_decay, learning_rate, num_epochs, batch_size, i = None):
    train_iter = load_array((X_train, y_train), batch_size)
    test_iter = load_array((X_valid, y_valid), batch_size, False)

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
    optimizer = torch.optim.SGD(net.parameters(), learning_rate, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss(reduction='none')

    def _init_weight(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(_init_weight)

    data_outcome, metric = [], Accumulator(3)
    # num_batches = len(train_iter)
    for epoch in range(num_epochs):
        net.train().to(device)
        # metric = Accumulator(3)
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad()
            y_hat = net(X).to(device)
            y = y.reshape(len(y)).long()
            l = loss(y_hat, y).to(device)
            l.backward()
            optimizer.step()

            # with torch.no_grad():
            #     metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            # train_metrics = metric[0] / metric[2], metric[1] / metric[2]
            # train_l, train_acc = train_metrics
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, None))
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())

        train_metrics = metric[0] / metric[2], metric[1] / metric[2]
        train_loss, train_acc = train_metrics
        test_acc = evaluate_accuracy(net, test_iter)

        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(f'epoch {epoch + 1}, train_loss {train_loss:f}, train_acc {train_acc:f}, test_acc {test_acc:f}')
        data_outcome.extend([train_loss, train_acc, test_acc])

        # conf_matrix, test_acc = torch.zeros(9, 9), []
        # with torch.no_grad():
        #     for X, y in test_iter:
        #         y = y.squeeze()
        #
        #         # 将变量转为gpu
        #         X.to(device)
        #         y.to(device)
        #
        #         y_hat = net(X)
        #
        #         # 记录混淆矩阵参数
        #         conf_matrix = dcm.confusion_matrix(y_hat, y, conf_matrix)
        #         conf_matrix = conf_matrix.cpu()
        #
        # labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        # dcm.plot_confusion_matrix(conf_matrix, 9, labels)

    # data_outcome = np.array(data_outcome).reshape(num_epochs, 3)
    # np.savetxt('D:/1.data/13_outcome/data_outcome3.csv', data_outcome, delimiter=',')


# def k_fold(k, net, features, labels, weight_decay, learning_rate, num_epochs, batch_size):
#     for i in range(k):
#         X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, features, labels)
#         train(net, X_train, y_train, X_valid, y_valid, weight_decay, learning_rate, num_epochs, batch_size, i)

k, weight_decay, lr, num_epochs, batch_size = 3, 0.015, 0.0003, 100, 128
# k_fold(k, net, features, labels, weight_decay, lr, num_epochs, batch_size)
train(net, X_train, y_train, X_test, y_test, weight_decay, lr, num_epochs, batch_size)
d2l.plt.show()