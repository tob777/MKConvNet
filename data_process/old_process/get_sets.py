import pandas as pd
import os

in_root = 'D:/3.data/2.data_unemg_lowb'
out_root1 = 'D:/3.data/3.data/conv_data_train_test_6'

out_root2 = 'D:/1.data/12_test_sets/conv_train_test_uf_6'

name = ['normal.csv', 'sandbag_2kg.csv', 'sandbag_4kg.csv', 'insole_1layer.csv', 'insole_3layer.csv', 'insole_2layer.csv', 'AFO_0°.csv', 'AFO_20°.csv', 'AFO_30°.csv']

in_file_list = os.listdir(in_root)
is_split = False

for i in range(0, 18, 2):
    tip, data_all = 0, pd.DataFrame()
    for root in in_file_list:
        files = os.listdir(in_root + '/' + root)
        print(root, files[i])
        data1 = pd.read_csv(in_root + '/' + root + '/' + files[i])
        data2 = pd.read_csv(in_root + '/' + root + '/' + files[i + 1])
        if tip == 0:
            data_all = pd.concat((data1, data2))
            tip += 1
        else:
            data_all = pd.concat((data_all, data1, data2))

    # if is_split:
    #     n_train = data_all.shape[0] // 4 * 3
    #     data_train = data_all[:n_train]
    #     data_test = data_all[n_train:]
    #
    #     os.makedirs(out_root1, exist_ok=True)
    #     os.makedirs(out_root2, exist_ok=True)
    #
    #     name_id = i // 2
    #     path_out1 = os.path.join(out_root1, name[name_id])
    #     path_out2 = os.path.join(out_root2, name[name_id])
    #     data_train.to_csv(path_out1, index=False)
    #     data_test.to_csv(path_out2, index=False)
    #
    # else:
    # os.makedirs(out_root1, exist_ok=True)

    name_id = i // 2
    path_out1 = os.path.join(out_root1, name[name_id])
    data_all.to_csv(path_out1, index=False)