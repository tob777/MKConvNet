import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def confusion_matrix(y_hat, y, conf_matrix):
    y_hat = torch.argmax(y_hat, 1).long()
    y = y.long()
    for p, t in zip(y_hat, y):
        conf_matrix[p, t] += 1
    return conf_matrix

import itertools
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, epoch, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        cm1 = cm
        print('Confusion matrix, without normalization')
    # print(cm)
    plt.figure(figsize=(14, 14))
    plt.imshow(cm1, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # plt.axis("equal")

    # ax = plt.gca()
    # left, right = plt.xlim()
    # ax.spines['left'].set_position(('data', left))
    # ax.spines['right'].set_position(('data', right))
    # for edge_i in ['top', 'bottom', 'right', 'left']:
    #     ax.spines[edge_i].set_edgecolor("white")

    # fmt = '.4f' if normalize else 'd'
    thresh = cm1.max() / 2.
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        num = '{:.4f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 horizontalalignment="center",  verticalalignment='center',
                 color="white" if cm1[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.savefig('D:/4.outcome/confusion_matrix/cm{}.png'.format(str(epoch)))
    plt.close()

    # calculate precision、 recall and F1-score
    precision = np.around((np.diagonal(cm) / cm.sum(axis=0)), 4).reshape(len(classes), 1)
    recall = np.around((np.diagonal(cm) / cm.sum(axis=1)), 4).reshape(len(classes), 1)
    F1_score = np.around(2 * (precision * recall) / (precision + recall), 4)
    cm = np.append(cm, precision, axis=1)
    cm = np.append(cm, recall, axis=1)
    cm = np.append(cm, F1_score, axis=1)

    # store confusion matrix to excel file
    excel_cm = pd.DataFrame(cm)
    excel_cm.index = classes
    classes.extend(['Recall', 'Precision', 'F1-score'])
    excel_cm.columns = classes
    writer = pd.ExcelWriter('D:/4.outcome/confusion_matrix/cm{}.xlsx'.format(str(epoch)))
    excel_cm.to_excel(writer)
    writer.save()

# def plot_confusion_matrix(conf_matrix, kinds, labels, epoch):
#
#     # 显示数据
#     plt.figure()
#     plt.imshow(conf_matrix, cmap=plt.cm.Blues)
#
#     # 在图中标注数量/概率信息
#     thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
#     for x in range(kinds):
#         for y in range(kinds):
#             # 注意这里的matrix[y, x]不是matrix[x, y]
#             info = int(conf_matrix[y, x])
#             plt.text(x, y, info,
#                      verticalalignment='center',
#                      horizontalalignment='center',
#                      color="white" if info > thresh else "black")
#
#     plt.tight_layout()  # 保证图不重叠
#     plt.yticks(range(kinds), labels)
#     plt.xticks(range(kinds), labels, rotation=45)  # X轴字体倾斜45°
#     plt.savefig('D:/4.outcome/confusion_matrix/cm{}.png'.format(str(epoch)))
#     plt.close()
