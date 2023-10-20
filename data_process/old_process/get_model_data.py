import pandas as pd
import os

in_root = 'D:/1.data/15_model_data'
out_root = 'D:/1.data/16_node_data'

files = os.listdir(in_root)
cnt = 31
len = len(files)
for i in range(len):
    if i == 0:
        train = pd.read_csv(in_root + '/' + files[i])
        train = train.iloc[:, 0: -1]
        train_post = pd.read_csv(in_root + '/' + files[i + 1])
        train = pd.concat((train.iloc[0:], train_post[0:]), axis=1)
        name = 'node' + str(cnt) + '.csv'
        train.to_csv(out_root + '/' + name, index=False)
    elif i == len - 1:
        train_pre = pd.read_csv(in_root + '/' + files[i - 1])
        train_pre = train_pre.iloc[:, 0: -1]
        train = pd.read_csv(in_root + '/' + files[i])
        train = pd.concat((train.iloc[0:], train_pre[0:]), axis=1)
        name = 'node' + str(cnt) + '.csv'
        trian.to_csv(out_root + '/' + name, index=False)
    else:
        train_pre = pd.read_csv(in_root + '/' + files[i - 1])
        train_pre = train_pre.iloc[:, 0: -1]
        train = pd.read_csv(in_root + '/' + files[i])
        train = train.iloc[:, 0: -1]
        train = pd.concat((train_pre.iloc[0:], train.iloc[0:]), axis=1)
        train_post = pd.read_csv(in_root + '/' + files[i + 1])
        trian = pd.concat((train.iloc[0:], train_post[0:]), axis=1)
        name = 'node' + str(cnt) + '.csv'
        trian.to_csv(out_root + '/' + name, index=False)
    cnt += 1