import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import os

path = "D:/3.data/1.raw_data/1.six/chen_yu_lin/下身骨科靴0度右脚1.csv"
# data = np.loadtxt("D:/3.data/1.raw_data/1.six/chen_yu_lin/下身骨科靴0度右脚1.csv", delimiter=',')
f = open(path)
data = pd.read_csv(f)

x = plt.figure(figsize=(8, 4)).add_subplot()
y = plt.figure(figsize=(8, 4)).add_subplot()
z = plt.figure(figsize=(8, 4)).add_subplot()

ankle_acc_x_vaxis = data.iloc[400:1000, 3].values
ankle_acc_y_vaxis = data.iloc[400:1000, 5].values
ankle_acc_z_vaxis = data.iloc[400:1000, 7].values
X = numpy.array([i for i in range(0, 601)])

x.plot(ankle_acc_x_vaxis, 'k')
y.plot(ankle_acc_y_vaxis, 'k')
z.plot(ankle_acc_z_vaxis, 'k')

x.set_xticks([])
x.set_yticks([])
y.set_xticks([])
y.set_yticks([])
z.set_xticks([])
z.set_yticks([])

plt.show()
