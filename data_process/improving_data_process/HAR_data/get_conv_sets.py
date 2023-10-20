import pandas as pd
import os

in_root = 'D:/3.data/2.data_unemg_lowb/3.twelve'
out_root1 = 'D:/3.data/3.conv_data/3.conv_data_train_test_12'

out_root2 = 'D:/1.data/12_test_sets/conv_train_test_uf_6'

name = ['normal.csv', 'sandbag_2kg.csv', 'sandbag_4kg.csv', 'insole_1layer.csv', 'insole_3layer.csv', 'insole_2layer.csv', 'AFO_0°.csv', 'AFO_20°.csv', 'AFO_30°.csv']

in_file_list = os.listdir(in_root)
is_split = False

for i in range(0, 18, 2):
    tip, data_all = 0, pd.DataFrame()
    num_data = 0
    for root in in_file_list:
        files = os.listdir(in_root + '/' + root)
        print(root, ' ', files[i], ' ', files[i + 1])
        data1 = pd.read_csv(in_root + '/' + root + '/' + files[i])
        data2 = pd.read_csv(in_root + '/' + root + '/' + files[i + 1])

        # normalization
        data1_cols = data1.columns
        data1 = data1[data1_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        data2_cols = data2.columns
        data2 = data2[data2_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        print(data1.shape)
        print(data2.shape)
        num_data = num_data + data1.shape[0] + data2.shape[0]
        if tip == 0:
            data_all = pd.concat((data1, data2))
            tip += 1
            if data_all.shape[0] != num_data:
                print('wrong')
                os.system('pause')
        else:
            data_all = pd.concat((data_all, data1, data2))
            if data_all.shape[0] != num_data:
                print('wrong')
                os.system('pause')
        print(data_all.shape[0])
        print(num_data)

    print(num_data)
    print(data_all.shape)
    data_all.drop_duplicates(subset=None, keep=False, inplace=True)
    print(data_all.shape)

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
    os.makedirs(out_root1, exist_ok=True)

    name_id = i // 2
    path_out1 = os.path.join(out_root1, name[name_id])
    data_all.to_csv(path_out1, index=False)