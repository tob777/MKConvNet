import os
import random

import torch
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data_root = 'D:/2.data/12_test_sets/conv_data_uf'
# window_size = 300
# batch_size = 64
def conv_data_iter(data_root, batch_size, window_size):
    files = os.listdir(data_root)
    data_labels = [6, 7, 8, 3, 4, 5, 0, 1, 2]
    indices = list(range(0, 9))
    random.shuffle(indices)

    feature_list, label_list = [], []
    for file_num in indices:
        file = files[file_num]
        label = data_labels[file_num]
        path = os.path.join(data_root, file)

        # load data and normalization
        data = pd.read_csv(path)
        cols = data.columns
        data = data[cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # get features
        data = torch.tensor(data.values, dtype=torch.float32).to(device)
        num_data = data.shape[0] // window_size
        data = data[0: num_data * window_size, :]
        data = data.reshape(num_data, window_size, data.shape[1])
        feature_list.append(data)

        # get label
        label = torch.full((num_data, 1), label, dtype=torch.long).to(device)
        label_list.append(label)

    data_all = torch.cat((feature_list[0], feature_list[1], feature_list[2], feature_list[3], feature_list[4],
                          feature_list[5], feature_list[6], feature_list[7], feature_list[8],)).to(device)
    label_all = torch.cat((label_list[0], label_list[1], label_list[2], label_list[3], label_list[4],
                           label_list[5], label_list[6], label_list[7], label_list[8])).to(device)

# print(data_all, ' ', label_all.shape)
    num_data = data_all.shape[0]
    index = list(range(num_data))
    random.shuffle(index)
    for i in range(0, num_data, batch_size):
        batch_indices = torch.tensor(index[i: min(i + batch_size, num_data)])
        # data_iter = data_all[i: min(i + batch_size, num_data), :]
        # label_iter = label_all[i: min(i + batch_size, num_data), :]
        yield data_all[batch_indices], label_all[batch_indices]

class conv_dataloader:
    def __init__(self, data_root, batch_size, window_size):
        self.data_iter = conv_data_iter
        self.data_root = data_root
        self.batch_size = batch_size
        self.window_size = window_size

    def __iter__(self):
        return self.data_iter(self.data_root, self.batch_size, self.window_size)