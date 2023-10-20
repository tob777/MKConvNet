import os
import pandas as pd

colum_all = [i for i in range(0, 84)]
colum_left = [3, 5, 7, 9, 11, 13, 31, 33, 35, 37, 39, 41, 59, 61, 63, 65, 67, 69]
colum_right = [17, 19, 21, 23, 25, 27, 45, 47, 49, 51, 53, 55, 73, 75, 77, 79, 81, 83]

#数据输入输出路径
subject_path = "D:/3.data/1.raw_data/5.twenty_one"
out_path = "D:/3.data/1.raw_data/10.left_right_data_21"

subject_names = os.listdir(subject_path)
for name in subject_names:
    files = os.listdir(os.path.join(subject_path, name))
    print(name)
    for file in files:
        print(file)
        f = open(os.path.join(subject_path, name, file))
        data = pd.read_csv(f)

        #取出左右下肢数据
        data.columns = colum_all
        data_left = data[colum_left]
        data_right = data[colum_right]

        #取出合适数据
        data_left = data_left[444:len(data_left.dropna(axis=0)) - 444]
        data_right = data_right[444:len(data_right.dropna(axis=0)) - 444]
        print(data_left.shape[0])
        print(data_right.shape[0])

        #保存数据
        store_path_left = os.path.join(out_path, name, "left")
        store_path_right = os.path.join(out_path, name, "right")
        os.makedirs(store_path_left, exist_ok=True)
        os.makedirs(store_path_right, exist_ok=True)
        data_left.to_csv(os.path.join(store_path_left, file), index=False)
        data_left.to_csv(os.path.join(store_path_right, file), index=False)

