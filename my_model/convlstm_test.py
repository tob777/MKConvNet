import warnings
import train_model_9_cls as tm
warnings.filterwarnings('ignore')

import random
import numpy as np
import sklearn.metrics as metrics

import torch
from torch import nn
import torch.nn.functional as F

NB_SENSOR_CHANNELS = 36


class HARModel(nn.Module):
    def __init__(self, n_hidden=128, n_layers=1, n_filters=64, n_classes=9, filter_size=5, drop_prob=0.5):
        super(HARModel, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size

        self.conv1 = nn.Conv1d(NB_SENSOR_CHANNELS, n_filters, filter_size)
        self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size)
        self.conv4 = nn.Conv1d(n_filters, n_filters, filter_size)

        self.lstm1 = nn.LSTM(n_filters, n_hidden, n_layers)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers)

        self.fc = nn.Linear(n_hidden, n_classes)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden, batch_size):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(8, -1, self.n_filters)
        x, hidden = self.lstm1(x, hidden)
        x, hidden = self.lstm2(x, hidden)

        x = x.contiguous().view(-1, self.n_hidden)
        x = self.dropout(x)
        x = self.fc(x)

        out = x.view(batch_size, -1, self.n_classes)[:, -1, :]

        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden

net = HARModel()

def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Conv1d or type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)
net.apply(init_weights)


train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')

def conv_data_iter(data_all, label_all, batch_size, is_shuffle=False):
    num_data = data_all.shape[0]
    index = list(range(num_data))

    if is_shuffle:
        random.shuffle(index)
    # print(index)

    for i in range(0, num_data, batch_size):
        if i + batch_size < num_data:
            batch_indices = torch.tensor(index[i: i + batch_size])
        # data_iter = data_all[i: min(i + batch_size, num_data), :]
        # label_iter = label_all[i: min(i + batch_size, num_data), :]
        yield data_all[batch_indices], label_all[batch_indices]

def load_conv_data(data_all, label_all, batch_size, is_shuffle=False):
    data_iter = conv_dataloader(data_all, label_all, batch_size, is_shuffle)
    return data_iter

class conv_dataloader:
    def __init__(self, data_all, label_all, batch_size, is_shuffle):
        self.data_iter = conv_data_iter
        self. data_all = data_all
        self.label_all = label_all
        self.batch_size = batch_size
        self.is_shuffle = is_shuffle

    def __iter__(self):
        return self.data_iter(self.data_all, self.label_all, self.batch_size, self.is_shuffle)

def train(net,train_data, train_label, X_test, y_test, epochs=20, batch_size=100, lr=0.008):
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_iter = load_conv_data(train_data, train_label, batch_size, is_shuffle=True)
    test_iter = load_conv_data(X_test, y_test, batch_size)
    if (train_on_gpu):
        net.cuda()

    for e in range(epochs):

        # initialize hidden state
        h = net.init_hidden(batch_size)
        # h = net.init_hidden(67)
        train_losses = []
        accuracy_train = 0
        net.train()
        for batch in train_iter:
            inputs, targets = batch
            inputs = inputs.permute(0, 2, 1)
            # inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            opt.zero_grad()

            # get the output from the model
            output, h = net(inputs, h, batch_size)

            loss = criterion(output, targets.reshape(len(targets)).long())
            train_losses.append(loss.item())
            loss.backward()
            opt.step()

            top_p, top_class = output.topk(1, dim=1)
            equals = top_class == targets.view(*top_class.shape).long()
            accuracy_train += torch.mean(equals.type(torch.FloatTensor))

        val_h = net.init_hidden(batch_size)
        val_losses = []
        accuracy_test = 0
        f1score = 0
        net.eval()
        with torch.no_grad():
            for batch in test_iter:
                inputs, targets = batch
                inputs = inputs.permute(0, 2, 1)
                # inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

                val_h = tuple([each.data for each in val_h])

                if (train_on_gpu):
                    inputs, targets = inputs.cuda(), targets.cuda()

                output, val_h = net(inputs, val_h, batch_size)

                val_loss = criterion(output, targets.reshape(len(targets)).long())
                val_losses.append(val_loss.item())

                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == targets.view(*top_class.shape).long()
                accuracy_test += torch.mean(equals.type(torch.FloatTensor))
                f1score += metrics.f1_score(top_class.cpu(), targets.view(*top_class.shape).long().cpu(),
                                            average='weighted')

        net.train()  # reset to train mode after iterationg through validation data

        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Train Loss: {:.4f}...".format(np.mean(train_losses)),
              "Train Acc: {:.4f}...".format(accuracy_train / (len(train_label) // batch_size)),
              "Val Loss: {:.4f}...".format(np.mean(val_losses)),
              "Val Acc: {:.4f}...".format(accuracy_test / (len(X_test) // batch_size)),
              "F1-Score: {:.4f}...".format(f1score / (len(X_test) // batch_size)))

SLIDING_WINDOW_LENGTH = 24
SLIDING_WINDOW_STEP = 12

data_root = 'D:/3.data/3.conv_data/1.conv_data_train_test_6'
train_data, train_label, test_data, test_label = tm.data_processing(data_root, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
train(net, train_data, train_label, test_data, test_label)