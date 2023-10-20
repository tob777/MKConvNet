import os
import pandas as pd
import torch
from torch_geometric.data import Data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_processing(data_root, window_size, step, num_kinds, data_labels):
    files = os.listdir(data_root)

    print('loading data ...')

    data_list = []
    edge_list = torch.tensor([[i for i in range(0, window_size - 1)], [i for i in range(1, window_size)]], dtype=torch.long).contiguous()


    # edge_list = torch.cat((torch.arange(0, window_size - 2), torch.arange(1, window_size - 1)), dim=1)

    for file_num in range(num_kinds):
        file = files[file_num]
        label = data_labels[file_num]
        path = os.path.join(data_root, file)

        # load_data 、 normalization and duplicate_removal
        data = pd.read_csv(path)
        data.drop_duplicates(subset=None, keep='first', inplace=True)

        # get graph
        data = torch.tensor(data.values, dtype=torch.float32)
        num_data = data.shape[0]

        for i in range(0, num_data, step):
            if i + window_size > num_data:
                break
            data_graph = Data(x=data[i:i + window_size].to(device), edge_index=edge_list.to(device), y=label)
            data_list.append(data_graph)

    return data_list