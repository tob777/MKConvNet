import os
import pandas as pd

in_root = 'D:/3.data/2.data_unemg_lowb/6.data_sym_AFO'
out_root1 = 'D:/3.data/3.conv_data/8.data_sym_AFO_12_train_test/train_data'
out_root2 = 'D:/3.data/3.conv_data/8.data_sym_AFO_12_train_test/test_data'

data_kinds = os.listdir(in_root)
colum = [i for i in range(0, 18)]
data_val = ['AFO_0°_abnorm', 'AFO_20°_abnorm', 'AFO_30°_abnorm', 'AFO_0°_norm', 'AFO_20°_norm', 'AFO_30°_norm']
name_cnt = 0

for data_kind in data_kinds:
    print(data_kind)
    object_name = os.listdir(os.path.join(in_root, data_kind))
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

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
                col1 = data1.columns
                data2.columns = colum
                col2 = data2.columns
                train_data = data1[col1]
                test_data = data2[col2]
            else:
                data1.columns = colum
                col1 = data1.columns
                data2.columns = colum
                col2 = data2.columns
                train_data = pd.concat((train_data, data1[col1]))
                test_data = pd.concat((test_data, data2[col2]))

        output_train = os.path.join(out_root1, data_val[name_cnt] + '.csv')
        output_test = os.path.join(out_root2, data_val[name_cnt] + '.csv')
        train_data.to_csv(output_train, index=False)
        test_data.to_csv(output_test, index=False)
        name_cnt += 1