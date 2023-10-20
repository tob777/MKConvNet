import os
import pandas as pd

in_root = 'D:/3.data/2.data_unemg_lowb/9.21_sym_no_norm_normal'
out_root1 = 'D:/3.data/3.conv_data/15.data_sym_norm_6_train_test_no_normlization/train_data/'
out_root2 = 'D:/3.data/3.conv_data/15.data_sym_norm_6_train_test_no_normlization/test_data/'
data_name = ['left.csv', 'right.csv']

object_names = os.listdir(in_root)

colum = [i for i in range(0, 84)]
colum_left = [3, 5, 7, 9, 11, 13, 31, 33, 35, 37, 39, 41, 59, 61, 63, 65, 67, 69]
colum_right = [17, 19, 21, 23, 25, 27, 45, 47, 49, 51, 53, 55, 73, 75, 77, 79, 81, 83]

train_data_left = pd.DataFrame()
train_data_right = pd.DataFrame()
test_data_left = pd.DataFrame()
test_data_right = pd.DataFrame()

for object_name in object_names:
    print(object_name)
    dir1 = os.path.join(in_root, object_name)
    file_names = os.listdir(dir1)
    for file_name in file_names:
        print(file_name)
        dir2 = os.path.join(dir1, file_name)
        f = open(dir2)
        data = pd.read_csv(f)

        data.columns = colum
        data = data[296:len(data.dropna(axis=0)) - 295]
        train_num = data.shape[0] // 4 * 3

        train = data[:train_num]
        test = data[train_num:]

        train_left = train[colum_left]
        print(train_left.shape)
        test_left = test[colum_left]
        train_right = train[colum_right]
        print(train_right.shape)
        test_right = train[colum_right]

        train_data_left = pd.concat((train_data_left, train_left))
        test_data_left = pd.concat((test_data_left, test_left))
        train_data_right = pd.concat((train_data_right, train_left))
        test_data_right = pd.concat((test_data_right, test_left))

train_data_left.to_csv(os.path.join(out_root1 + data_name[0]), index=False)
train_data_right.to_csv(os.path.join(out_root1 + data_name[1]), index=False)
test_data_left.to_csv(os.path.join(out_root2 + data_name[0]), index=False)
test_data_right.to_csv(os.path.join(out_root2 + data_name[1]), index=False)
