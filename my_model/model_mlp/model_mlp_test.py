import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils import data
from d2l import torch as d2l
from torch.nn import functional as F
import os
import draw_conf_matrix as dcm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data = pd.read_csv('D:/1.data/11_train_sets/train_sets_unemg.csv')
# train_data2 = pd.read_csv('D:/2.data/11_train_sets/train.csv')
test_data = pd.read_csv('D:/1.data/12_test_sets/test_sets.csv')

train_data = train_data.dropna(axis=0, how='any')



# train_data = pd.concat((train_data1.iloc[:, 0:], train_data2.iloc[:, 0:]))
all_features = pd.concat((train_data.iloc[:, 0:-1], test_data.iloc[:, 0:-1]))
col = all_features.columns
all_features[col] = all_features[col].apply(lambda x: (x - x.mean()) / (x.std()))
all_features = all_features.dropna(axis=0, how='any')

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_data.label.values.reshape(-1, 1), dtype=torch.float32).to(device)
test_labels = torch.tensor(test_data.label.values.reshape(-1, 1), dtype=torch.float32).to(device)

loss = nn.CrossEntropyLoss(reduction='none')
in_features = train_features.shape[1]

def load_array(train_features, test_features):
    train_features = train_features.reshape(train_features.shape[0], 1, 1, train_features.shape[1])
    test_features = test_features.reshape(test_features.shape[0], 1, test_features.shape[1])
    return train_features, test_features

class get_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(1, 32, 2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 1),
            nn.Conv1d(32, 48, 2, stride=1),
        )

        self.linear_layer = nn.Sequential(
            nn.Linear(48, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    def forward(self, X):
        X = self.conv_layer(X)
        X = X.view(-1, 48)
        X = self.linear_layer(X)

        return X

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

    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]

def train(net, train_features, train_labels, test_features, test_labels, weight_decay, learning_rate,  num_epoch, batch_size, i):
    train_iter, test_iter = load_array(train_features, test_features)
    y = train_labels.reshape(train_labels.shape[0], 1)
    # test_l = test_labels

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 1.4], legend=['train loss', 'train acc', 'test acc'])
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    data_outcome = []
    metric = Accumulator(3)
    for epoch in range(num_epoch):
        net.train().to(device)
        # metric = Accumulator(3)
        for X in train_iter:
            optimizer.zero_grad()
            y_hat = net(X).to(device)
            y = y.reshape(len(y)).long()
            l = loss(y_hat, y).to(device)
            l.mean().backward()
            optimizer.step()
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
            train_metrics = metric[0] / metric[2], metric[1] / metric[2]
            train_loss, train_acc = train_metrics

        # test_acc = evaluate_accuracy(net, test_iter)
        # animator.add(epoch+1, train_metrics + (test_acc,))
        print(f'epoch {epoch + 1}, train_loss {train_loss:f}, train_acc {train_acc:f}') #, test_acc {test_acc}')
        # data_outcome.extend([train_loss, train_acc, test_acc])

        # conf_matrix, test_acc = torch.zeros(4, 4), []
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
        # labels = [0, 1, 2, 3]
        # dcm.plot_confusion_matrix(conf_matrix, 4, labels)

    data_outcome = np.array(data_outcome).reshape(num_epoch, 3)
    np.savetxt('D:/1.data/13_outcome/data_outcome{}.csv'.format(i), data_outcome, delimiter=',')


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

def k_fold(k, X_train, y_train, weight_decay, learning_rate, num_epochs, batch_size):
    # for i in range(k):
    #     data = get_k_fold_data(k, i, X_train, y_train)
    #     net = get_net().to(device)
    #     train(net, *data, weight_decay, learning_rate, num_epochs, batch_size, i)
    net = get_net().to(device)
    train(net, X_train, y_train, test_features, test_labels, weight_decay, learning_rate, num_epochs, batch_size, i = 1)
    # test_iter = load_array((test_features, test_labels), 256)

    # conf_matrix = torch.zeros(4, 4)
    # with torch.no_grad():
    #     for X, y in test_iter:
    #         y = y.squeeze()
    #
    #         # 将变量转为gpu
    #         X.to(device)
    #         y.to(device)
    #
    #         y_hat = net(X)
    #         # 记录混淆矩阵参数
    #         conf_matrix = dcm.confusion_matrix(y_hat, y, conf_matrix)
    #         conf_matrix = conf_matrix.cpu()
    #
    # labels = [0, 1, 2, 3]
    # dcm.plot_confusion_matrix(conf_matrix, 4, labels)

k, weight_decay, lr, num_epochs, batch_size = 6, 0, 0.01, 10, 32
k_fold(k, train_features, train_labels, weight_decay, lr, num_epochs, batch_size)
d2l.plt.show()