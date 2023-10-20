import os
import pandas as pd

AFO_path = "D:/3.data/1.raw_data/6.AFO"
sandbag_path = "D:/3.data/1.raw_data/7.sandbag"
insole_path = "D:/3.data/1.raw_data/8.insole"
normal_path = "D:/3.data/1.raw_data/9.normal"

paths = [insole_path, AFO_path, sandbag_path,  normal_path]

colum_all = [i for i in range(0, 84)]
colum = [3, 5, 7, 9, 11, 13, 17, 19, 21, 23, 25, 27, 31, 33, 35, 37, 39, 41, 45, 47, 49, 51, 53, 55, 59, 61, 63, 65, 67, 69, 73, 75, 77, 79, 81, 83]

#遍历所有类型数据路径
for path in paths:
    object_names = os.listdir(path)
    data1_train = pd.DataFrame()
    data2_train = pd.DataFrame()
    data3_train = pd.DataFrame()
    data1_test = pd.DataFrame()
    data2_test = pd.DataFrame()
    data3_test = pd.DataFrame()
    #遍历所有受试者步态数据
    for object_name in object_names:
        print(object_name)
        files = os.listdir(os.path.join(path, object_name))
        i = -1
        #划分训练集与测试集
        for file in files:
            i += 1
            print(file)
            f = open(os.path.join(path, object_name, file))
            data = pd.read_csv(f)

            #取出所需数据
            data.columns = colum_all
            data = data[colum]
            data = data[444:len(data.dropna(axis=0)) - 444]

            #划分比例
            data_split = data.shape[0] // 10 * 7
            data_train = data[:data_split]
            data_test = data[data_split:]

            #将数据按类型连接
            if i < 2:
                data1_train = pd.concat((data1_train, data_train))
                data1_test = pd.concat((data1_test, data_test))
            elif i >= 2 and i < 4:
                data2_train = pd.concat((data2_train, data_train))
                data2_test = pd.concat((data2_test, data_test))
            else:
                data3_train = pd.concat((data3_train, data_train))
                data3_test = pd.concat((data3_test, data_test))
    #存储数据
    if path == AFO_path:
        AFO_train_outPath = "D:/3.data/4.data_set/AFO/train_set"
        AFO_test_outPath = "D:/3.data/4.data_set/AFO/test_set"
        AFO_name = ["AFO_0°.csv", "AFO_20°.csv", "AFO_30°.csv"]
        data1_train.to_csv(os.path.join(AFO_train_outPath, AFO_name[0]), index=False)
        data1_test.to_csv(os.path.join(AFO_test_outPath, AFO_name[0]), index=False)
        data2_train.to_csv(os.path.join(AFO_train_outPath, AFO_name[1]), index=False)
        data2_test.to_csv(os.path.join(AFO_test_outPath, AFO_name[1]), index=False)
        data3_train.to_csv(os.path.join(AFO_train_outPath, AFO_name[2]), index=False)
        data3_test.to_csv(os.path.join(AFO_test_outPath, AFO_name[2]), index=False)
    elif path == sandbag_path:
        sandbag_train_outPath = "D:/3.data/4.data_set/sandbag/train_set"
        sandbag_test_outPath = "D:/3.data/4.data_set/sandbag/test_set"
        sandbag_name = ["sandbag_2kg.csv", "sandbag_4kg.csv"]
        data1_train.to_csv(os.path.join(sandbag_train_outPath, sandbag_name[0]), index=False)
        data1_test.to_csv(os.path.join(sandbag_test_outPath, sandbag_name[0]), index=False)
        data2_train.to_csv(os.path.join(sandbag_train_outPath, sandbag_name[1]), index=False)
        data2_test.to_csv(os.path.join(sandbag_test_outPath, sandbag_name[1]), index=False)
    elif path == insole_path:
        insole_train_outPath = "D:/3.data/4.data_set/insole/train_set"
        insole_test_outPath = "D:/3.data/4.data_set/insole/test_set"
        insole_name = ["insole_1l.csv", "insole_3l.csv", "insole_2l.csv"]
        data1_train.to_csv(os.path.join(insole_train_outPath, insole_name[0]), index=False)
        data1_test.to_csv(os.path.join(insole_test_outPath, insole_name[0]), index=False)
        data2_train.to_csv(os.path.join(insole_train_outPath, insole_name[1]), index=False)
        data2_test.to_csv(os.path.join(insole_test_outPath, insole_name[1]), index=False)
        data3_train.to_csv(os.path.join(insole_train_outPath, insole_name[2]), index=False)
        data3_test.to_csv(os.path.join(insole_test_outPath, insole_name[2]), index=False)
    else:
        normal_train_outPath = "D:/3.data/4.data_set/normal/train_set"
        normal_test_outPath = "D:/3.data/4.data_set/normal/test_set"
        normal_name = "normal.csv"
        data1_train.to_csv(os.path.join(normal_train_outPath, normal_name), index=False)
        data1_test.to_csv(os.path.join(normal_test_outPath, normal_name), index=False)



