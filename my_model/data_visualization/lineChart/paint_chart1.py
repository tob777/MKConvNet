import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("D:/3.data/3.conv_data/12.data_sym_all_21_train_test/train_data/AFO_0°_abnorm.csv", delimiter=',')
ankle_acc_x_vaxis = data[1:451, 0]
ankle_acc_y_vaxis = data[1:451, 1]
ankle_acc_z_vaxis = data[1:451, 2]
ankle_gro_x_vaxis = data[1:451, 3]
ankle_gro_y_vaxis = data[1:451, 4]
ankle_gro_z_vaxis = data[1:451, 5]
knee_acc_x_vaxis = data[1:451, 6]
knee_acc_y_vaxis = data[1:451, 7]
knee_acc_z_vaxis = data[1:451, 8]
knee_gro_x_vaxis = data[1:451, 9]
knee_gro_y_vaxis = data[1:451, 10]
knee_gro_z_vaxis = data[1:451, 11]
hip_acc_x_vaxis = data[1:451, 12]
hip_acc_y_vaxis = data[1:451, 13]
hip_acc_z_vaxis = data[1:451, 14]
hip_gro_x_vaxis = data[1:451, 15]
hip_gro_y_vaxis = data[1:451, 16]
hip_gro_z_vaxis = data[1:451, 17]

fig, axes = plt.subplots(18, 1, figsize=(100, 20), sharex=True, sharey=True)

plt.subplots_adjust(wspace=0, hspace=0)


axes[0].plot(ankle_acc_x_vaxis, "brown", label="x_ankle_acc")
axes[1].plot(ankle_acc_y_vaxis, "k", label="y_ankle_acc")
axes[2].plot(ankle_acc_z_vaxis, "peru", label="z_ankle_acc")
axes[3].plot(ankle_gro_x_vaxis, "darkorange", label="x_ankle_gro")
axes[4].plot(ankle_gro_y_vaxis, "yellowgreen", label="y_ankle_gro")
axes[5].plot(ankle_gro_z_vaxis, "deepskyblue", label="z_ankle_gro")
axes[6].plot(knee_acc_x_vaxis, "brown", label="x_knee_acc")
axes[7].plot(knee_acc_y_vaxis, "k", label="y_knee_acc")
axes[8].plot(knee_acc_z_vaxis, "peru", label="z_knee_acc")
axes[9].plot(knee_gro_x_vaxis, "darkorange", label="x_knee_gro")
axes[10].plot(knee_gro_y_vaxis, "yellowgreen", label="y_knee_gro")
axes[11].plot(knee_gro_z_vaxis, "deepskyblue", label="z_knee_gro")
axes[12].plot(hip_acc_x_vaxis, "brown", label="x_hip_acc")
axes[13].plot(hip_acc_y_vaxis, "k", label="y_hip_acc")
axes[14].plot(hip_acc_z_vaxis, "peru", label="z_hip_acc")
axes[15].plot(hip_gro_x_vaxis, "darkorange", label="x_hip_gro")
axes[16].plot(hip_gro_y_vaxis, "yellowgreen", label="y_hip_gro")
axes[17].plot(hip_gro_z_vaxis, "deepskyblue", label="z_hip_gro")

axes[0].legend(loc=5)
axes[1].legend(loc=5)
axes[2].legend(loc=5)
axes[3].legend(loc=5)
axes[4].legend(loc=5)
axes[5].legend(loc=5)
axes[6].legend(loc=5)
axes[7].legend(loc=5)
axes[8].legend(loc=5)
axes[9].legend(loc=5)
axes[10].legend(loc=5)
axes[11].legend(loc=5)
axes[12].legend(loc=5)
axes[13].legend(loc=5)
axes[14].legend(loc=5)
axes[15].legend(loc=5)
axes[16].legend(loc=5)
axes[17].legend(loc=5)


axes[0].set_xticks([])
axes[0].set_yticks([])
plt.xlabel("Time Series")

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()