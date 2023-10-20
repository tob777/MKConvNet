import torch
import matplotlib.pyplot as plt
import numpy as np

def confusion_matrix(y_hat, y, conf_matrix):
    y_hat = torch.argmax(y_hat, 1).long()
    y = y.long()
    for p, t in zip(y_hat, y):
        conf_matrix[p, t] += 1
    return conf_matrix


# import itertools
# # 绘制混淆矩阵
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     Input
#     - cm : 计算出的混淆矩阵的值
#     - classes : 混淆矩阵中每一行每一列对应的列
#     - normalize : True:显示百分比, False:显示个数
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     print(cm)
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
#         plt.text(j, i, num,
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')

def plot_confusion_matrix(conf_matrix, kinds, labels):

    # 显示数据
    plt.figure()
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(kinds):
        for y in range(kinds):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(kinds), labels)
    plt.xticks(range(kinds), labels, rotation=45)  # X轴字体倾斜45°

