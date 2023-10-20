import os
import pandas as pd

in_root = 'D:/3.data/2.data_unemg_lowb/5.data_sym_global_norm/5.data_sym_AFO_test_2'
out_root = 'D:/3.data/3.conv_data/11.data_sym_AFo_12_train_test_other_person/test_data'

max = [12.1948, 7.2466, 14.4536, 1193.7200, 1180.5500, 1302.4400,
       7.4385, 4.9287, 6.0596, 591.2200, 1587.6800, 692.8050,
       2.4673, 3.4697, 3.2798, 347.0730, 400.3050, 265.3050]
min = [-9.4819, -10.3481, -7.9531, -932.1950, -1450.5500, -834.4510,
       -6.1519, -8.5440, -3.7471, -814.6950, -1107.2600, -825.4880,
       -4.1958, -5.5474, -2.2983, -322.6220, -385.7930, -203.4760]

data_kinds = os.listdir(in_root)
colum = [i for i in range(0, 18)]
data_val = ['AFO_0°_abnorm', 'AFO_20°_abnorm', 'AFO_30°_abnorm', 'AFO_0°_norm', 'AFO_20°_norm', 'AFO_30°_norm']
name_cnt = 0

for data_kind in data_kinds:
    print(data_kind)
    object_name = os.listdir(os.path.join(in_root, data_kind))
    all_data = pd.DataFrame()

    for i in range(0, 6, 2):
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
                all_data = pd.concat((data1, data2))
            else:
                data1.columns = colum
                data2.columns = colum
                all_data = pd.concat((all_data, data1, data2))

        for i in range(all_data.shape[1]):
            all_data[i] = (all_data[i] - min[i]) / (max[i] - min[i])

        output = os.path.join(out_root, data_val[name_cnt] + '.csv')
        all_data.to_csv(output, index=False)
        name_cnt += 1