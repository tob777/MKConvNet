import os
import pandas as pd

in_root = 'D:/3.data/2.data_unemg_lowb/6.data_sym_AFO'
out_root = 'D:/3.data/3.conv_data/6.data_sym_AFO_12'

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

        output = os.path.join(out_root, data_val[name_cnt] + '.csv')
        all_data.to_csv(output, index=False)
        name_cnt += 1