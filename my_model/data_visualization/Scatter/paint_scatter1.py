import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import DecisionBoundaryDisplay

points = pd.read_excel("D:/4.outcome/painting/normal/out.xlsx")
y = pd.read_excel("D:/4.outcome/painting/normal/y.xlsx")

# points = pd.read_excel("D:/4.outcome/painting/AFO_0°/out.xlsx")
# y = pd.read_excel("D:/4.outcome/painting/AFO_0°/y.xlsx")

points = points.to_numpy()
y = y.to_numpy()

X = points[:, 1: 3]
X1 = X[0:217]
X2 = X[217:417]
X3 = X[417:617]
X4 = X[617:-1]
X = X[300: -1]

y = y[:, 1]
y1 = y[0:217]
y2 = y[217:417]
y3 = y[417:617]
y4 = y[617:-1]
y = y[300: -1]

#异常步态
# X = points[:, 1: 3]
# X1 = X[0: 505]
# X2 = X[505: -1]


#异常步态
# y = y[:, 1]
# y1 = y[0: 505]
# y2 = y[505: -1]

cm_bright = ListedColormap(["#FF0000", "#0000FF"])
#
# clf = make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.025))
# clf.fit(data, y)
# DecisionBoundaryDisplay.from_estimator(clf, data, cmap=plt.cm.RdBu, alpha=0.8, eps=0.5)
#
# plt.scatter(data[:, 0], data[:, 1], c=y, cmap=cm_bright, edgecolors="k")
# plt.xticks([])
# plt.yticks([])

def plot_decision_boundary(pred_func) :
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    #用预测函徵预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    #然后画出图
    #plt.cm.brg
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.4)

    plt.scatter(X1[:, 0], X1[:, 1], c=y1, label="Left Gait", cmap=ListedColormap("#FF0000"), edgecolors="k")
    plt.scatter(X3[:, 0], X3[:, 1], c=y3, label="Right Gait", cmap=ListedColormap("#0000FF"), edgecolors="k")
    plt.scatter(X4[:, 0], X4[:, 1], c=y4,  cmap=ListedColormap("#0000FF"), edgecolors="k")
    plt.scatter(X2[:, 0], X2[:, 1], c=y2, alpha=0.3, cmap=ListedColormap("#FF0000"), edgecolors="k")

    #异常步态
    # plt.scatter(X1[:, 0], X1[:, 1], c=y1, label="Impaired Gait", cmap=ListedColormap("#FF0000"), edgecolors="k")
    # plt.scatter(X2[:, 0], X2[:, 1], c=y2, label="Noraml Gait", cmap=ListedColormap("#0000FF"), edgecolors="k")

    plt.xticks([])
    plt.yticks([])
    plt.legend(prop=FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14))


from sklearn.linear_model import LogisticRegressionCV
#训练逻辑回归分苑器
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X, y)
#画一下决策边界
plot_decision_boundary(lambda x: clf.predict(x))
plt.title('', fontproperties='SimHei', fontsize=15)

plt.show()