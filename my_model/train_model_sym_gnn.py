import torch
from torch_geometric.loader import DataLoader
from torch import nn
import numpy as np
from d2l import torch as d2l
import my_model.draw_conf_matrix as dcm
from my_model.model_gnn_sym.sage_conv.SageConv_1_2 import my_sageCN
# from my_model.model_gnn_sym.gcn_model.gcn_sym import my_GCN
from my_model.model_gnn_sym.data_recon.gcn_data_pro import data_processing

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

    # cmp = d2l.astype(y_hat, y.dtype) == y

    # return float(d2l.astype(cmp, y.dtype).sum())
    return (y_hat == y).sum().item()

def evaluate_accuracy(net, data_iter, epoch, max_test_acc):
    """Compute the accuracy for a model on a dataset.

    Defined in :numref:`sec_softmax_scratch`"""
    # if isinstance(net, torch.nn.Module):
    net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    conf_matrix = torch.zeros(6, 6)
    with torch.no_grad():
        for X in data_iter:
            # y = y.squeeze()

            # # 将变量转为gpu
            X.to(device)
            X.y.to(device)

            y_hat = net(X).to(device)

            test_acc = accuracy(y_hat, X.y)
            y_size = X.y.numel()

            metric.add(test_acc, y_size)

            #记录混淆矩阵参数
            conf_matrix = dcm.confusion_matrix(y_hat, X.y, conf_matrix)
            conf_matrix = conf_matrix.cpu()

    labels = ['AFO_0°_norm', 'AFO_0°_abnorm', 'AFO_20°_norm', 'AFO_20°_abnorm', 'AFO_30°_norm', 'AFO_30°_abnorm']
    if metric[0] / metric[1] > max_test_acc:
        dcm.plot_confusion_matrix(conf_matrix, labels, epoch)
    return metric[0] / metric[1]

loss = nn.CrossEntropyLoss()

def train(net, train_data, test_data, lr, weight_decay, num_epochs, batch_size, is_shuffle, lr_period, lr_decay, is_lr_decay=False):
    train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=is_shuffle)
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    net.to(device)

    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.0, 1.5], legend=['train loss', 'train acc', 'test acc'])
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    if is_lr_decay:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)

    data_outcome, max_test_acc = [], 0
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, X, in enumerate(train_iter):
            optimizer.zero_grad()
            y_hat = net(X).to(device)
            X.y = X.y.to(device)
            l = loss(y_hat, X.y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l.sum() * y_hat.shape[0], accuracy(y_hat, X.y), X.y.numel())

            # train_l = metric[0] / metric[2]
            # train_acc = metric[1] / metric[2]
            # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
            #     test_acc = evaluate_accuracy(net, test_iter, epoch + (i + 1) / num_batches)
            #     animator.add(epoch + (i + 1) / num_batches, (train_l, train_acc, test_acc))
        if is_lr_decay:
            scheduler.step()

        train_loss, train_acc = metric[0] / metric[2], metric[1] / metric[2]
        test_acc = evaluate_accuracy(net, test_iter, epoch, max_test_acc)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
        # animator.add(epoch + 1, (None, None, test_acc))
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
        print(f'epoch {epoch + 1}, train_loss {train_loss:f}, train_acc {train_acc:f}, test_acc {test_acc:f}')
        data_outcome.extend([train_loss, train_acc, test_acc])

        # if train_acc > 0.95:
        #     patience += 1
        #     if patience > 2:
        #         torch.save(net, 'D:/1_study/programming_lauguage_learning/pythonProject/save_model_data/net/net_resCNN.pth')
        #         torch.save(net.state_dict(), 'D:/1_study/programming_lauguage_learning/pythonProject/save_model_data/net/net_resCNN_state.pth')
        #         break

    data_outcome = np.array(data_outcome).reshape(num_epochs, 3)
    np.savetxt('D:/4.outcome/2.sym_cls/gnn/sageconv/sageConv_1_2/out5.csv', data_outcome, delimiter=',')

#torch.load('D:/1_study/programming_lauguage_learning/pythonProject/save_model_data/net/net_CNN.pth')
# net.load_state_dict(torch.load('D:/1_study/programming_lauguage_learning/pythonProject/save_model_data/net/net_CNN_state.pth'))

label_all, label_4, label_sym_AFO = [6, 7, 8, 3, 4, 5, 0, 1, 2], [1, 2, 3, 0], [1, 0, 3, 2, 5, 4]
num_kinds = 6
window_size_train, step_train = 150, 50
window_size_test, step_test = 150, 150
lr, num_epochs, batch_size, is_shuffle = 0.001, 512, 32, True
weight_decay, lr_period, lr_decay, is_lr_decay = 0.00001, 20, 0.9, True

# x = torch.rand(1, 36, 150)
# y = net(x)
# print(y.shape)

train_data_root = 'D:/3.data/3.conv_data/9.data_sym_AFO_12_train_test_glo/train_data'
test_data_root = 'D:/3.data/3.conv_data/9.data_sym_AFO_12_train_test_glo/test_data'
train_data = data_processing(train_data_root, window_size_train, step_train, num_kinds, label_sym_AFO)
test_data = data_processing(test_data_root, window_size_test, step_test, num_kinds, label_sym_AFO)
# for i in range(train_data.shape[0]):
#     for j in range(test_data.shape[0]):
#         print(train_data[i] == test_data[j])
#         if train_data[i][0][0] == test_data[j][0][0]:
#             os.system('pause')

# net = my_model.model_gnn_sym.ggcn_model.test_ggcn_1_1.test_GGCN(batch_size)
net = my_sageCN()

train(net, train_data, test_data, lr, weight_decay, num_epochs, batch_size, is_shuffle, lr_period, lr_decay, is_lr_decay)
d2l.plt.show()