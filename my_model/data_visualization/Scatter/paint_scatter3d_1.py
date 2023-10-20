import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from umap import UMAP
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties


points = pd.read_excel("D:/4.outcome/painting/8cls/out.xlsx")
Y = pd.read_excel("D:/4.outcome/painting/8cls/y.xlsx")

points = points.to_numpy()
Y = Y.to_numpy()

X = points[:, 1:9]
Y = Y[:, 1]

#LDA降维
# lda = LinearDiscriminantAnalysis(n_components=3)
# X_lda = lda.fit_transform(X, Y)

# PCA降维
# pca = PCA(n_components=3)
# X = pca.fit_transform(X)

#t-SNE降维
# tsne = TSNE(n_components=3, perplexity=30.0)
# X_tsne = tsne.fit_transform(X)

#UMAP降维
umap_model = UMAP(n_components=3, n_neighbors=50, min_dist=0.1)
X = umap_model.fit_transform(X)

x = X[:, 0]
y = X[:, 1]
z = X[:, 2]

# 生成3个分类的数据
x1 = x[100:200]
y1 = y[100:200]
z1 = z[100:200]
c1 = Y[100:200]

x2 = x[700:800]
y2 = y[700:800]
z2 = z[700:800]
c2 = Y[700:800]

x3 = x[1200:1300]
y3 = y[1200:1300]
z3 = z[1200:1300]
c3 = Y[1200:1300]
         
x4 = x[1600:1700]
y4 = y[1600:1700]
z4 = z[1600:1700]
c4 = Y[1600:1700]

x5 = x[2000:2100]
y5 = y[2000:2100]
z5 = z[2000:2100]
c5 = Y[2000:2100]

x6 = x[2500:2600]
y6 = y[2500:2600]
z6 = z[2500:2600]
c6 = Y[2500:2600]

x7 = x[2900:3000]
y7 = y[2900:3000]
z7 = z[2900:3000]
c7 = Y[2900:3000]

x8 = x[3400:3500]
y8 = y[3400:3500]
z8 = z[3400:3500]
c8 = Y[3400:3500]

# 创建3D图形对象
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'y', 'c', 'm', 'grey', 'orange']   # 设置不同分类的颜色
labels = ['Orthopedic Boots 0' + '$\degree$' + ' Normal', 'Orthopedic Boots 20' + '$\degree$'  + ' Normal',
          'Orthopedic Boots 30' + '$\degree$'  + ' Normal',
          'Height-increasing Shoe 3cm Normal', 'Height-increasing Shoe 4.5cm Normal', 'Height-increasing Shoe 6cm Normal',
          'Sandbag 2kg Normal', 'Sandbag 4kg Normal']
# 绘制散点图
ax.scatter(x1, y1, z1, c=colors[0], label=labels[0], edgecolors="k")#, marker='o'   colors[0]    c1  , cmap='rainbow'
ax.scatter(x2, y2, z2, c=colors[1], label=labels[1], edgecolors="k")#, marker='^'   colors[1]    c2  , cmap='rainbow'
ax.scatter(x3, y3, z3, c=colors[2], label=labels[2], edgecolors="k")#, marker='s'   colors[2]    c3  , cmap='rainbow'
ax.scatter(x4, y4, z4, c=colors[3], label=labels[3], edgecolors="k")#, marker='s'   colors[3]    c4  , cmap='rainbow'
ax.scatter(x5, y5, z5, c=colors[4], label=labels[4], edgecolors="k")#, marker='s'   colors[4]    c5  , cmap='rainbow'
ax.scatter(x6, y6, z6, c=colors[5], label=labels[5], edgecolors="k")#, marker='s'   colors[5]    c6  , cmap='rainbow'
ax.scatter(x7, y7, z7, c=colors[6], label=labels[6], edgecolors="k")#, marker='s'   colors[6]    c7  , cmap='rainbow'
ax.scatter(x8, y8, z8, c=colors[7], label=labels[7], edgecolors="k")#, marker='s'   colors[7]    c8  , cmap='rainbow'

# 设置坐标轴标签
ax.set_xlabel('X', fontproperties='SimHei', fontsize=12)
ax.set_ylabel('Y', fontproperties='SimHei', fontsize=12)
ax.set_zlabel('Z', fontproperties='SimHei', fontsize=12)

ax.legend(prop=FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=10), loc='upper left')

# 显示图形
plt.show()