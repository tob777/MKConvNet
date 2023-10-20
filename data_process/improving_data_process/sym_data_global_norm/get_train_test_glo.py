import os
import pandas as pd

in_root = 'D:/3.data/2.data_unemg_lowb/7.data_sym_global_norm_21/1.data_split_glo'
out_root1 = 'D:/3.data/3.conv_data/12.data_sym_all_21_train_test/train_data'
out_root2 = 'D:/3.data/3.conv_data/12.data_sym_all_21_train_test/test_data'

data_kinds = os.listdir(in_root)
colum = [i for i in range(0, 18)]
data_val = ['AFO_0°_abnorm', 'AFO_20°_abnorm', 'AFO_30°_abnorm', 'AFO_0°_norm', 'AFO_20°_norm', 'AFO_30°_norm',
            'insole_1l_abnorm', 'insole_3l_abnorm', 'insole_2l_abnorm', 'insole_1l_norm', 'insole_3l_norm', 'insole_2l_norm',
            'normal_left', 'normal_right',
            'sandbag_2kg_abnorm', 'sandbag_4kg_abnorm', 'sandbag_2kg_norm', 'sandbag_4kg_norm']
train_data_all = []
test_data_all = []
name_cnt = 0

for data_kind in data_kinds:
    print(data_kind)
    object_name = os.listdir(os.path.join(in_root, data_kind))
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    length = len(os.listdir(os.path.join(in_root, data_kind, object_name[0])))
    for i in range(0, length, 2):
        tip = 0
        for name in object_name:
            print(name)
            file_name_path = os.path.join(in_root, data_kind, name)
            file_name = os.listdir(file_name_path)

            file1, file2 = file_name[i], file_name[i + 1]
            data1 = pd.read_csv(os.path.join(file_name_path, file1))
            data2 = pd.read_csv(os.path.join(file_name_path, file2))
            print(file1, ' ', file2)
            if tip == 0:
                tip = 1
                data1.columns = colum
                data2.columns = colum
                num1 = data1.shape[0] // 4 * 3
                num2 = data2.shape[0] // 4 * 3
                train_data = pd.concat((data1[:num1], data2[:num2]))
                test_data = pd.concat((data1[num1:], data2[num2:]))
            else:
                data1.columns = colum
                data2.columns = colum
                num1 = data1.shape[0] // 4 * 3
                num2 = data2.shape[0] // 4 * 3
                train_data = pd.concat((train_data, data1[:num1], data2[:num2]))
                test_data = pd.concat((test_data, data1[num1:], data2[num2:]))

        train_data_all.append(train_data)
        test_data_all.append(test_data)

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


for i in range(0, len(data_count) - 1, 2):
    train_data = data_all[data_count[i]:data_count[i + 1]]
    test_data = data_all[data_count[i + 1]:data_count[i + 2]]

    output_train = os.path.join(out_root1, data_val[name_cnt] + '.csv')
    output_test = os.path.join(out_root2, data_val[name_cnt] + '.csv')
    train_data.to_csv(output_train, index=False)
    test_data.to_csv(output_test, index=False)
    name_cnt += 1