import torch
from torch import nn
import numpy as np
from d2l import torch as d2l
import cnn_dataloader
import draw_conf_matrix as dcm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    conf_matrix = torch.zeros(9, 9)
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

            metric.add(accuracy(y_hat, y), d2l.size(y))

            #记录混淆矩阵参数
            conf_matrix = dcm.confusion_matrix(y_hat, y, conf_matrix)
            conf_matrix = conf_matrix.cpu()

    labels = ['normal', 'sandbag_2kg', 'sandbag_4kg', 'insole_1layer', 'insole_2layer', 'insole_3layer', 'AFO_0°', 'AFO_20°', 'AFO_30°']
    dcm.plot_confusion_matrix(conf_matrix, 9, labels, epoch)
    return metric[0] / metric[1]

def load_conv_data(data_root, batch_size, window_size):
    data_iter = cnn_dataloader.conv_dataloader(data_root, batch_size, window_size)
    return data_iter

net = nn.Sequential(
    nn.Conv1d(36, 48, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool1d(3, 2),
    nn.Conv1d(48, 128, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool1d(3, 2),
    nn.Conv1d(128, 192, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv1d(192, 192, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv1d(192, 128, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool1d(3, 2),
    nn.Flatten(),
    nn.Linear(5 * 128, 2048), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(2048, 2048), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(2048, 9)
)

X = torch.rand((2, 36, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)

# def train(net, train_set_root, test_set_root, lr, num_epochs, batch_size, window_size):
#     train_iter = load_conv_data(train_set_root, batch_size, window_size)
#     test_iter = load_conv_data(test_set_root, batch_size, window_size)
#
#     def _init_weight(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv1d:
#             nn.init.xavier_uniform_(m.weight)
#     net.apply(_init_weight)
#     net.to(device)
#
#     animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['train loss', 'train acc', 'test acc'])
#     optimizer = torch.optim.Adam(net.parameters(), lr)
#     loss = nn.CrossEntropyLoss(reduction='none')
#
#     data_outcome, metric = [], Accumulator(3)
#     for epoch in range(num_epochs):
#         net.train().to(device)
#         for X, y in train_iter:
#             optimizer.zero_grad()
#             X = X.permute(0, 2, 1).to(device)
#             y_hat = net(X).to(device)
#             y = y.reshape(len(y))
#             l = loss(y_hat, y).to(device)
#             l.mean().backward()
#             optimizer.step()
#
#             metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
#
#         train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
#         test_acc = evaluate_accuracy(net, test_iter, epoch)
#         animator.add(epoch + 1, (train_loss, train_acc, test_acc))
#         print(f'epoch {epoch + 1}, train_loss {train_loss:f}, train_acc {train_acc:f}, test_acc {test_acc:f}')
#         data_outcome.extend([train_loss, train_acc, test_acc])
#
#     data_outcome = np.array(data_outcome).reshape(num_epochs, 3)
#     np.savetxt('D:/1.data/13_outcome/1uf_data_outcome.csv', data_outcome, delimiter=',')
#
# train_root = 'D:/2.data/11_train_sets/conv_data_uf'
# test_root = 'D:/2.data/12_test_sets/conv_data_uf'
#
# lr, num_epochs, batch_size, window_size = 0.000003, 300, 256, 224
# train(net, train_root, test_root, lr, num_epochs, batch_size, window_size)
# d2l.plt.show()