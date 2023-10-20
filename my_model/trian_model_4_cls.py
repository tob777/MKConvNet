import torch
from torch import nn
import numpy as np
from my_model.model_cnn import model_cnn as cnet
from d2l import torch as d2l
import my_model.draw_conf_matrix as dcm
import os
import random
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_processing(data_root, window_size, step, num_kinds, data_labels, is_shuffle=False):
    files = os.listdir(data_root)
    indices = list(range(0, num_kinds))
    if is_shuffle:
        random.shuffle(indices)

    print('loading data ...')

    feature_list, label_list = [], []
    for file_num in indices:
        file = files[file_num]
        label = data_labels[file_num]
        path = os.path.join(data_root, file)

        # load_data 、 normalization and duplicate_removal
        data = pd.read_csv(path)
        data.drop_duplicates(subset=None, keep='first', inplace=True)
        cols = data.columns
        data = data[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # get features
        data = torch.tensor(data.values, dtype=torch.float32).to(device)
        num_data = data.shape[0]
        data = data.unsqueeze(dim=0)
        data_section = data[:, 0: window_size]

        for i in range(step, num_data, step):
            if i + window_size > num_data:
                break
            data_frag = data[:, i:i + window_size]
            data_section = torch.cat((data_section, data_frag), dim=0)

        # data_section = torch.tensor(data.values, dtype=torch.float32).to(device)
        # num_data_section = data_section.shape[0] // window_size
        # data_section = data_section[:num_data_section * window_size]
        # data_section = data_section.reshape(num_data_section, window_size, data_section.shape[1])
        #
        # for num_step in range(step, window_size, step):
        #     data_frag = torch.tensor(data[num_step:].values, dtype=torch.float32).to(device)
        #     num_data_frag = data_frag.shape[0] // window_size
        #     data_frag = data_frag[:num_data_frag * window_size]
        #     data_frag = data_frag.reshape(num_data_frag, window_size, data_frag.shape[1])
        #     data_section = torch.cat((data_section, data_frag))

        # get label
        num_data = data_section.shape[0]
        label = torch.full((num_data, 1), label, dtype=torch.long).to(device)
        label_list.append(label)

        feature_list.append(data_section)

    data_all = torch.cat((feature_list[0], feature_list[1]))
    label_all = torch.cat((label_list[0], label_list[1]))
    for i in range(2, num_kinds):
        data_all = torch.cat((data_all, feature_list[i]))
        label_all = torch.cat((label_all, label_list[i]))

    data_all.to(device)
    label_all.to(device)

    indics_all = list(range(data_all.shape[0]))
    random.shuffle(indics_all)
    train_num = data_all.shape[0] // 2

    train_data = data_all[indics_all[:train_num]]
    train_label = label_all[indics_all[:train_num]]
    test_data = data_all[indics_all[train_num:]]
    test_label = label_all[indics_all[train_num:]]
    # print(train_data == test_data)

    return train_data, train_label, test_data, test_label


def conv_data_iter(data_all, label_all, batch_size, is_shuffle=False):
    num_data = data_all.shape[0]
    index = list(range(num_data))

    if is_shuffle:
        random.shuffle(index)
    # print(index)

    for i in range(0, num_data, batch_size):
        batch_indices = torch.tensor(index[i: min(i + batch_size, num_data)])
        # data_iter = data_all[i: min(i + batch_size, num_data), :]
        # label_iter = label_all[i: min(i + batch_size, num_data), :]
        yield data_all[batch_indices], label_all[batch_indices]

class conv_dataloader:
    def __init__(self, data_all, label_all, batch_size, is_shuffle):
        self.data_iter = conv_data_iter
        self. data_all = data_all
        self.label_all = label_all
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle

    def __iter__(self):
        return self.data_iter(self.data_all, self.label_all, self.batch_size, self.is_shuffle)


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

    cmp = d2l.astype(y_hat, y.dtype) == y

    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter, epoch):
    """Compute the accuracy for a model on a dataset.

    Defined in :numref:`sec_softmax_scratch`"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    conf_matrix = torch.zeros(4, 4)
    with torch.no_grad():
        for X, y in data_iter:
            # y = y.squeeze()
            #
            # # 将变量转为gpu
            X = X.permute(0, 2, 1)
            X.to(device)
            y.to(device)

            y_hat = net(X).to(device)
            y = y.reshape(len(y))

            test_acc = accuracy(y_hat, y)
            y_size = d2l.size(y)

            metric.add(test_acc, y_size)

            #记录混淆矩阵参数
            conf_matrix = dcm.confusion_matrix(y_hat, y, conf_matrix)
            conf_matrix = conf_matrix.cpu()

    labels = ['normal', 'AFO_0°', 'AFO_20°', 'AFO_30°']
    dcm.plot_confusion_matrix(conf_matrix, 4, labels, epoch)
    return metric[0] / metric[1]

def load_conv_data(data_all, label_all, batch_size, is_shuffle=False):
    data_iter = conv_dataloader(data_all, label_all, batch_size, is_shuffle)
    return data_iter


loss = nn.CrossEntropyLoss()

def train(net, train_data, train_label, test_data, test_label, lr, weight_decay, num_epochs, batch_size, is_shuffle, lr_period, lr_decay, is_lr_decay=False):
    train_iter = load_conv_data(train_data, train_label, batch_size, is_shuffle=is_shuffle)
    test_iter = load_conv_data(test_data, test_label, batch_size)

    def _init_weight(m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(_init_weight)
    net.to(device)

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 3], legend=['train loss', 'train acc', 'test acc'])
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    if is_lr_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)

    patience = 0
    data_outcome = []
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train().to(device)
        for X, y in train_iter:
            optimizer.zero_grad()
            X = X.permute(0, 2, 1).to(device)
            y_hat = net(X).to(device)
            y = y.reshape(len(y))
            l = loss(y_hat, y).to(device)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l.sum() * y_hat.shape[0], accuracy(y_hat, y), y.numel())
        if is_lr_decay:
            scheduler.step()

        train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter, epoch)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
        print(f'epoch {epoch + 1}, train_loss {train_loss:f}, train_acc {train_acc:f}, test_acc {test_acc:f}')
        data_outcome.extend([train_loss, train_acc, test_acc])

        # if train_acc > 0.95:
        #     patience += 1
        #     if patience > 2:
        #         torch.save(net, 'D:/1_study/programming_lauguage_learning/pythonProject/save_model_data/net/net_resCNN.pth')
        #         torch.save(net.state_dict(), 'D:/1_study/programming_lauguage_learning/pythonProject/save_model_data/net/net_resCNN_state.pth')
        #         break


    # data_outcome = np.array(data_outcome).reshape(num_epochs, 3)
    # np.savetxt('D:/4.outcome/AFO_data/3uf_data_outcome.csv', data_outcome, delimiter=',')

net = cnet.my_cnn_model()#torch.load('D:/1_study/programming_lauguage_learning/pythonProject/save_model_data/net/net_CNN.pth')
# net.load_state_dict(torch.load('D:/1_study/programming_lauguage_learning/pythonProject/save_model_data/net/net_CNN_state.pth'))

label_all, label_4 = [6, 7, 8, 3, 4, 5, 0, 1, 2], [1, 2, 3, 0]
window_size, step, num_kinds,  = 150, 150, 4
lr, num_epochs, batch_size, is_shuffle = 0.0002, 80, 128, True
weight_decay, lr_period, lr_decay, is_lr_decay = 0.00001, 10, 0.9, False

# x = torch.rand(1, 36, 150)
# y = net(x)
# print(y.shape)

data_root = 'D:/3.data/3.conv_data/4.conv_data_train_test_AFO_6'
train_data, train_label, test_data, test_label = data_processing(data_root, window_size, step, num_kinds, label_4)

train(net, train_data, train_label, test_data, test_label, lr, weight_decay, num_epochs, batch_size, is_shuffle, lr_period, lr_decay, is_lr_decay)
d2l.plt.show()