import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

path = "D:/3.data/1.raw_data/3.twelve/chen_yu_lin/下身骨科靴0度右脚1.csv"
f = open(path)
data = pd.read_csv(f)

ankle_acc_x_vaxis = data.iloc[400:501, 3].values
ankle_acc_y_vaxis = data.iloc[400:501, 5].values
ankle_acc_z_vaxis = data.iloc[400:501, 7].values
ankle_gro_x_vaxis = data.iloc[400:501, 9].values
ankle_gro_y_vaxis = data.iloc[400:501, 11].values
ankle_gro_z_vaxis = data.iloc[400:501, 13].values
x = range(len(ankle_acc_x_vaxis))

fig, axes = plt.subplots(2, 1, figsize=(9, 9))
plt.subplots_adjust(hspace=0.3)
axes[0].tick_params(axis='both', which='major', labelsize=14)
axes[1].tick_params(axis='both', which='major', labelsize=14)

axes[0].plot(x, ankle_acc_x_vaxis, "red", label="x axis")
axes[0].plot(x, ankle_acc_y_vaxis, "darkorange", label="y axis")
axes[0].plot(x, ankle_acc_z_vaxis, "blue", label="z axis")
axes[0].grid(True)
axes[0].set_xlim(0, 100)
axes[0].set_xlabel('Sample points', fontproperties='SimHei', fontsize=20)
axes[0].set_ylabel('Acceleration/('+ '$m/s^2$'+ ')', fontproperties='SimHei', fontsize=20)

axes[1].plot(ankle_gro_x_vaxis, "red", label="x axis")
axes[1].plot(ankle_gro_y_vaxis, "darkorange", label="y axis")
axes[1].plot(ankle_gro_z_vaxis, "blue", label="z axis")
axes[1].grid(True)
axes[1].set_xlim(0, 100)
axes[1].set_xlabel('Sample points', fontproperties='SimHei', fontsize=20)
axes[1].set_ylabel('Angular velocity/(' + '$\degree/s$' + ')', fontproperties='SimHei', fontsize=20)

axes[0].legend(loc=0, prop=FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=17))
axes[1].legend(loc=0, prop=FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=17))

plt.show()