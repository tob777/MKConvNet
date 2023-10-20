#首先声明我是Python3.7版本#第一步要做的是导入一些头文件import importlib.util
import numpy as np
import struct
import sklearn
from sklearn import*
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
# #首先生成兹据,并将其绘恻出来最后一句是用来绘剩生成的救据集np.random.seed(e)
# X, y = sklearn.datasets.make_moons(200, noise=0.20)
#
# plt.scatter(X[ :, 0], X[:, 1], s=48, c=y, cmap=plt.cm.Spectral)
#
# #第二步绘悯决策边界
# def plot_decision_boundary(pred_func) :
#     x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
#     y_min, y_max = X[:, 1].min() - .5, X[:, 1].max()+ .5
#     h = 0.01
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#     #用预测函徵预测一下
#     Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     #然后画出图
#     plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
#     plt.scatter(X[ :, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
#
# from sklearn.linear_model import LogisticRegressionCV
# #训练逻辑回归分苑器
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X, y)
# #画一下决策边界
# plot_decision_boundary(lambda x: clf.predict(x))
# plt.title("Logistic Regression")
# plt.show()

# 生成数据
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 30, 40, 50]
# labels = ['A', 'B', 'C', 'D', 'E']
#
# # 绘制散点图并添加标签
# plt.scatter(x, y, label='Data Points')
# for i, label in enumerate(labels):
#     plt.annotate(label, (x[i], y[i]))
#
# plt.legend()
# plt.show()

# 生成三类数据并绘制散点图
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

# generate data
x1 = np.random.randn(100)
y1 = np.random.randn(100)
z1 = np.random.randn(100)

x2 = np.random.randn(100)+5
y2 = np.random.randn(100)+5
z2 = np.random.randn(100)+5

s1 = ax.scatter(x1, y1, z1, label="Data 1", s=50, alpha=0.8)
s2 = ax.scatter(x2, y2, z2, label="Data 2", s=50, alpha=0.8)

handles, labels = ax.get_legend_handles_labels()

# 设置 Legend 的位置
ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5))

plt.show()
