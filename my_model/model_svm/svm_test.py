import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# data_test = pd.read_csv(data_root)
# num = data_test.shape[0] // 150
# data_test = data_test[0: num * 150]
# data_id = np.array([i for i in range(0, num)]).repeat(150).reshape(-1, 1)
# data_time_series = np.tile([i for i in range(0, 150)], num).reshape(-1, 1)
# data_test.insert(0, "id", data_id)
# data_test.insert(1, "time_series", data_time_series)
# mean = data_test.mean().values
# var = data_test.var().values
# skew = data_test.skew().values
# k = np.concatenate((mean, var, skew))
# k1 = np.expand_dims(k, axis=0)
# k2 = np.expand_dims(k, axis=0)
# kk = np.concatenate((k1, k2))
# print(kk)
# print(kk.shape[0])

#print(data_test[0:150].mean().values) #, " ", data_test[0:150].var(), " ", data_test[0:150].skew())

# all type labels
# labels = ['normal',
#           'AFO_0°_norm', 'AFO_0°_abnorm', 'AFO_20°_norm', 'AFO_20°_abnorm', 'AFO_30°_norm', 'AFO_30°_abnorm',
#           'Insole_1l_norm', 'Insole_1l_abnorm', 'Insole_2l_norm', 'Insole_2l_abnorm', 'Insole_3l_norm', 'Insole_3l_abnorm',
#           'Sandbag_2kg_norm', 'Sandbag_2kg_abnorm', 'Sandbag_4kg_norm', 'Sandbag_4kg_abnorm']

# AFO type labels
# labels = ['AFO_norm', 'AFO_abnorm']
# labels = ['AFO_0°_norm', 'AFO_0°_abnorm', 'AFO_20°_norm', 'AFO_20°_abnorm', 'AFO_30°_norm', 'AFO_30°_abnorm']

# insole type labels
# labels = ['Insole_norm', 'Insole_abnorm']
# labels = ['Insole_1l_norm', 'Insole_1l_abnorm', 'Insole_2l_norm', 'Insole_2l_abnorm', 'Insole_3l_norm', 'Insole_3l_abnorm']

# sandbag type labels
# labels = ['Sandbag_norm', 'Sandbag_abnorm']
# labels = ['Sandbag_2kg_norm', 'Sandbag_2kg_abnorm', 'Sandbag_4kg_norm', 'Sandbag_4kg_abnorm']

# all type labels
labels = ['Noraml', 'Abnormal']

# 2cls single labels
# labels = ['AFO_0°_norm', 'AFO_0°_abnorm']
# labels = ['AFO_20°_norm', 'AFO_20°_abnorm']
# labels = ['AFO_30°_norm', 'AFO_30°_abnorm']
# labels = ['Insole_1l_norm', 'Insole_1l_abnorm']
# labels = ['Insole_2l_norm', 'Insole_2l_abnorm']
# labels = ['Insole_3l_norm', 'Insole_3l_abnorm']
# labels = ['Sandbag_2kg_norm', 'Sandbag_2kg_abnorm']
# labels = ['Sandbag_4kg_norm', 'Sandbag_4kg_abnorm']
# labels = ['normal_right', 'normal_left']

#8cls labels
# labels = ['AFO_0°_norm', 'AFO_20°_norm', 'AFO_30°_norm', 'insole_1l_norm', 'insole_2l_norm',
#           'insole_3l_norm', 'sandbag_2kg_norm', 'sandbag_4kg_norm']

def data_processing(data_root, window_size, step, num_kinds, data_labels, is_shuffle=False):
    files = os.listdir(data_root)
    indices = list(range(0, num_kinds))
    if is_shuffle:
        random.shuffle(indices)

    print('loading data ...')

    feature_list, label_list = [], []
    for file_num in indices:
        file = files[file_num]
        label = data_labels[file_num]
        path = os.path.join(data_root, file)

        # load_data 、 normalization and duplicate_removal
        data = pd.read_csv(path)
        data.drop_duplicates(subset=None, keep='first', inplace=True)

        # get features
        num_data = data.shape[0]
        data_section = data[0: window_size]
        mean = data_section.mean().values
        var = data_section.var().values
        skew = data_section.skew().values
        data_section = np.concatenate((mean, var, skew))
        data_section = np.expand_dims(data_section, axis=0)

        for i in range(step, num_data, step):
            if i + window_size > num_data:
                break
            data_frag = data[i:i + window_size]
            mean = data_frag.mean().values
            var = data_frag.var().values
            skew = data_frag.skew().values
            data_frag = np.concatenate((mean, var, skew))
            data_frag = np.expand_dims(data_frag, axis=0)
            data_section = np.concatenate((data_section, data_frag))

        # get label
        num_data = data_section.shape[0]
        label = np.full(num_data, label)
        label_list.append(label)
        feature_list.append(data_section)

    data_all = np.concatenate((feature_list[0], feature_list[1]))
    label_all = np.concatenate((label_list[0], label_list[1]))
    for i in range(2, num_kinds):
        data_all = np.concatenate((data_all, feature_list[i]))
        label_all = np.concatenate((label_all, label_list[i]))

    return data_all, label_all

label_2cls_all, label_all = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], \
                            [2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 0, 0, 14, 13, 16, 15]
label_2cls_AFO, label_sym_AFO = [1, 0, 1, 0, 1, 0], [1, 0, 3, 2, 5, 4]
label_2cls_insole, label_sym_insole = [1, 0, 1, 0, 1, 0], [1, 0, 3, 2, 5, 4]
label_2cls_sandbag, label_sym_sandbag = [1, 0, 1, 0], [1, 0, 3, 2]
label_2cls = [1, 0]
label_8cls = [0, 1, 2, 3, 4, 5, 6, 7]

num_data = 2
window_size_train, step_train = 150, 150
window_size_test, step_test = 150, 150

train_data_root = 'D:/3.data/3.conv_data/15.data_sym_norm_6_train_test_no_normlization/train_data'
test_data_root = 'D:/3.data/3.conv_data/15.data_sym_norm_6_train_test_no_normlization/test_data'
train_data, train_label = data_processing(train_data_root, window_size_train, step_train, num_data, label_2cls)
test_data, test_label = data_processing(test_data_root, window_size_test, step_test, num_data, label_2cls)

import sklearn
from sklearn.metrics import accuracy_score, classification_report
import sklearn.svm as svm

train_data, train_label = sklearn.utils.shuffle(train_data, train_label)
test_data, test_label = sklearn.utils.shuffle(test_data, test_label)

# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# dfss = ['ovo', 'ovr']
# cls_reports = []
test_accs = []

# for dfs in dfss:
#     for k in kernels:
#         model = svm.SVC(kernel=k, decision_function_shape=dfs)
#         model.fit(train_data, train_label)
#
#         y = model.predict(test_data)
#         test_acc = accuracy_score(test_label, y)
#         cls_report = classification_report(test_label, y, labels=label_8cls, target_names=labels, digits=4)
#
#         test_accs.append(test_acc)
#         cls_reports.append(cls_report)

model = svm.SVC(kernel='rbf')#, decision_function_shape='ovo')
model.fit(train_data, train_label)

y = model.predict(test_data)
test_acc = accuracy_score(test_label, y)
cls_report = pd.DataFrame(classification_report(test_label, y, target_names=labels, digits=4, output_dict=True)).T
test_accs.append(test_acc)

cls_report.to_csv('D:/4.outcome/2.sym_cls/cls_report1.csv')
test_accs = np.array(test_accs)
np.savetxt('D:/4.outcome/2.sym_cls/out1.csv', test_accs, delimiter=',')

print(test_accs)
print(cls_report)