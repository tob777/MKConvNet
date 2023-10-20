import os
import pandas as pd


colum_all = [i for i in range(0, 84)]
colum_left = [3, 5, 7, 9, 11, 13, 31, 33, 35, 37, 39, 41, 59, 61, 63, 65, 67, 69]
colum_right = [17, 19, 21, 23, 25, 27, 45, 47, 49, 51, 53, 55, 73, 75, 77, 79, 81, 83]

root_path = "D:/3.data/1.raw_data/11.symmetry_data"
paths = os.listdir(root_path)

#遍历所有类型数据路径
for path in paths:
    #存储各类型数据
    data1_train = pd.DataFrame()
    data2_train = pd.DataFrame()
    data3_train = pd.DataFrame()
    data1_test = pd.DataFrame()
    data2_test = pd.DataFrame()
    data3_test = pd.DataFrame()

    #遍历所有受试者步态数据
    object_names = os.listdir(os.path.join(root_path, path))
    for object_name in object_names:
        print(object_name)
        files = os.listdir(os.path.join(root_path, path, object_name))
        i = -1
        #划分训练集与测试集
        for file in files:
            i += 1
            print(file)
            f = open(os.path.join(root_path, path, object_name, file))
            data = pd.read_csv(f)

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
    train_outPath = 'D:/3.data/5.symmetry_gait_data_set/all_data/train_set'
    test_outPath = 'D:/3.data/5.symmetry_gait_data_set/all_data/test_set'
    if path == 'AFO_abnormal':
        AFO_name = ["AFO_0°_abnorm.csv", "AFO_20°_abnorm.csv", "AFO_30°_abnorm.csv"]
        data1_train.to_csv(os.path.join(train_outPath, AFO_name[0]), index=False)
        data1_test.to_csv(os.path.join(test_outPath, AFO_name[0]), index=False)
        data2_train.to_csv(os.path.join(train_outPath, AFO_name[1]), index=False)
        data2_test.to_csv(os.path.join(test_outPath, AFO_name[1]), index=False)
        data3_train.to_csv(os.path.join(train_outPath, AFO_name[2]), index=False)
        data3_test.to_csv(os.path.join(test_outPath, AFO_name[2]), index=False)
    elif path == 'AFO_normal':
        AFO_name = ["AFO_0°_norm.csv", "AFO_20°_norm.csv", "AFO_30°_norm.csv"]
        data1_train.to_csv(os.path.join(train_outPath, AFO_name[0]), index=False)
        data1_test.to_csv(os.path.join(test_outPath, AFO_name[0]), index=False)
        data2_train.to_csv(os.path.join(train_outPath, AFO_name[1]), index=False)
        data2_test.to_csv(os.path.join(test_outPath, AFO_name[1]), index=False)
        data3_train.to_csv(os.path.join(train_outPath, AFO_name[2]), index=False)
        data3_test.to_csv(os.path.join(test_outPath, AFO_name[2]), index=False)
    elif path == 'sandbag_abnormal':
        sandbag_name = ["sandbag_2kg_abnorm.csv", "sandbag_4kg_aborm.csv"]
        data1_train.to_csv(os.path.join(train_outPath, sandbag_name[0]), index=False)
        data1_test.to_csv(os.path.join(test_outPath, sandbag_name[0]), index=False)
        data2_train.to_csv(os.path.join(train_outPath, sandbag_name[1]), index=False)
        data2_test.to_csv(os.path.join(test_outPath, sandbag_name[1]), index=False)
    elif path == 'sandbag_normal':
        sandbag_name = ["sandbag_2kg_norm.csv", "sandbag_4kg_norm.csv"]
        data1_train.to_csv(os.path.join(train_outPath, sandbag_name[0]), index=False)
        data1_test.to_csv(os.path.join(test_outPath, sandbag_name[0]), index=False)
        data2_train.to_csv(os.path.join(train_outPath, sandbag_name[1]), index=False)
        data2_test.to_csv(os.path.join(test_outPath, sandbag_name[1]), index=False)
    elif path == 'insole_abnormal':
        insole_name = ["insole_1l_abnorm.csv", "insole_3l_abnorm.csv", "insole_2l_abnorm.csv"]
        data1_train.to_csv(os.path.join(train_outPath, insole_name[0]), index=False)
        data1_test.to_csv(os.path.join(test_outPath, insole_name[0]), index=False)
        data2_train.to_csv(os.path.join(train_outPath, insole_name[1]), index=False)
        data2_test.to_csv(os.path.join(test_outPath, insole_name[1]), index=False)
        data3_train.to_csv(os.path.join(train_outPath, insole_name[2]), index=False)
        data3_test.to_csv(os.path.join(test_outPath, insole_name[2]), index=False)
    elif path == 'insole_normal':
        insole_name = ["insole_1l_norm.csv", "insole_3l_norm.csv", "insole_2l_norm.csv"]
        data1_train.to_csv(os.path.join(train_outPath, insole_name[0]), index=False)
        data1_test.to_csv(os.path.join(test_outPath, insole_name[0]), index=False)
        data2_train.to_csv(os.path.join(train_outPath, insole_name[1]), index=False)
        data2_test.to_csv(os.path.join(test_outPath, insole_name[1]), index=False)
        data3_train.to_csv(os.path.join(train_outPath, insole_name[2]), index=False)
        data3_test.to_csv(os.path.join(test_outPath, insole_name[2]), index=False)
    elif path == 'normal_left':
        normal_name = "normal_left.csv"
        data1_train.to_csv(os.path.join(train_outPath, normal_name), index=False)
        data1_test.to_csv(os.path.join(test_outPath, normal_name), index=False)
    else:
        normal_name = "normal_right.csv"
        data1_train.to_csv(os.path.join(train_outPath, normal_name), index=False)
        data1_test.to_csv(os.path.join(test_outPath, normal_name), index=False)



