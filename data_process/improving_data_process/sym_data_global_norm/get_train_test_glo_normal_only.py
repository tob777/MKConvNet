import os
import pandas as pd

in_root = "D:/3.data/2.data_unemg_lowb/8.data_sym_global_norm_21_normal"

file_kinds = os.listdir(in_root)
out_root1 = "D:/3.data/3.conv_data/13.data_sym_21_train_test_norma_only/train_data"
out_root2 = "D:/3.data/3.conv_data/13.data_sym_21_train_test_norma_only/test_data"

data_val = ['normal_left', 'normal_right']

colum = [i for i in range(0, 18)]
train_data_all = []
test_data_all = []

for kind in file_kinds:
    persons = os.listdir(os.path.join(in_root, kind))
    print(kind)
    train_set = []
    test_set = []
    for person in persons:
        print(person)
        files = os.listdir(os.path.join(in_root, kind, person))
        for file in files:
            file_path = os.path.join(in_root, kind, person, file)
            print(file)

            data = pd.read_csv(file_path)
            data.columns = colum
            ratio = data.shape[0] // 4 * 3
            train_data = data[:ratio]
            test_data = data[ratio:]
            train_set.append(train_data)
            test_set.append(test_data)

    train = pd.concat((train_set[0], train_set[1]))
    test = pd.concat((test_set[0], test_set[1]))
    for i in range(2, len(train_set)):
        train = pd.concat((train, train_set[i]))
        test = pd.concat((test, test_set[i]))

    train_data_all.append(train)
    test_data_all.append(test)

data_count = [0]
data_count.append(train_data_all[0].shape[0])
data_count.append(data_count[-1] + test_data_all[0].shape[0])
data_all = pd.concat((train_data_all[0], test_data_all[0]))

for i in range(1, len(train_data_all)):
    data_count.append(data_count[-1] + train_data_all[i].shape[0])
    data_count.append(data_count[-1] + test_data_all[i].shape[0])
    data_all = pd.concat((data_all, train_data_all[i], test_data_all[i]))

data_all_col = data_all.columns
# print(data_all[data_all_col].max())
# print(data_all[data_all_col].min())

data_all = data_all[data_all_col].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

name_cnt = 0
for i in range(0, len(data_count) - 1, 2):
    train_data = data_all[data_count[i]:data_count[i + 1]]
    test_data = data_all[data_count[i + 1]:data_count[i + 2]]

    output_train = os.path.join(out_root1, data_val[name_cnt] + '.csv')
    output_test = os.path.join(out_root2, data_val[name_cnt] + '.csv')
    train_data.to_csv(output_train, index=False)
    test_data.to_csv(output_test, index=False)
    name_cnt += 1