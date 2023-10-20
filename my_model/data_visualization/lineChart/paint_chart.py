import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("D:/3.data/3.conv_data/12.data_sym_all_21_train_test/train_data/AFO_0°_abnorm.csv", delimiter=',')

fig, axes = plt.subplots(18, 1, sharex=True, sharey=True)

chart_ankle_acc_x = plt.figure(figsize=(10, 4)).add_subplot()
chart_ankle_acc_y = plt.figure(figsize=(10, 4)).add_subplot()
chart_ankle_acc_z = plt.figure(figsize=(10, 4)).add_subplot()
chart_ankle_gro_x = plt.figure(figsize=(10, 4)).add_subplot()
chart_ankle_gro_y = plt.figure(figsize=(10, 4)).add_subplot()
chart_ankle_gro_z = plt.figure(figsize=(10, 4)).add_subplot()
chart_knee_acc_x = plt.figure(figsize=(10, 4)).add_subplot()
chart_knee_acc_y = plt.figure(figsize=(10, 4)).add_subplot()
chart_knee_acc_z = plt.figure(figsize=(10, 4)).add_subplot()
chart_knee_gro_x = plt.figure(figsize=(10, 4)).add_subplot()
chart_knee_gro_y = plt.figure(figsize=(10, 4)).add_subplot()
chart_knee_gro_z = plt.figure(figsize=(10, 4)).add_subplot()
chart_hip_acc_x = plt.figure(figsize=(10, 4)).add_subplot()
chart_hip_acc_y = plt.figure(figsize=(10, 4)).add_subplot()
chart_hip_acc_z = plt.figure(figsize=(10, 4)).add_subplot()
chart_hip_gro_x = plt.figure(figsize=(10, 4)).add_subplot()
chart_hip_gro_y = plt.figure(figsize=(10, 4)).add_subplot()
chart_hip_gro_z = plt.figure(figsize=(10, 4)).add_subplot()

# ankle_acc_haxis = [i / 100 for i in range(0, 1500)]

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

chart_ankle_acc_x.plot(ankle_acc_x_vaxis, "brown", label="x_ankle_acc")
chart_ankle_acc_y.plot(ankle_acc_y_vaxis, "k", label="y_ankle_acc")
chart_ankle_acc_z.plot(ankle_acc_z_vaxis, "k", label="y_ankle_acc")
chart_ankle_gro_x.plot(ankle_gro_x_vaxis, "darkorange", label="x_ankle_gro")
chart_ankle_gro_y.plot(ankle_gro_y_vaxis, "yellowgreen", label="y_ankle_gro")
chart_ankle_gro_z.plot(ankle_gro_z_vaxis, "deepskyblue", label="z_ankle_gro")
chart_knee_acc_x.plot(knee_acc_x_vaxis, "brown", label="x_ankle_acc")
chart_knee_acc_y.plot(knee_acc_y_vaxis, "k", label="y_ankle_acc")
chart_knee_acc_z.plot(knee_acc_z_vaxis, "peru", label="z_ankle_acc")
chart_knee_gro_x.plot(knee_gro_x_vaxis, "darkorange", label="x_ankle_gro")
chart_knee_gro_y.plot(knee_gro_y_vaxis, "yellowgreen", label="y_ankle_gro")
chart_knee_gro_z.plot(knee_gro_y_vaxis, "deepskyblue", label="z_ankle_gro")
chart_hip_acc_x.plot(hip_acc_x_vaxis, "brown", label="x_ankle_acc")
chart_hip_acc_y.plot(hip_acc_y_vaxis, "k", label="y_ankle_acc")
chart_hip_acc_z.plot(hip_acc_z_vaxis, "peru", label="z_ankle_acc")
chart_hip_gro_x.plot(hip_gro_x_vaxis, "darkorange", label="x_ankle_gro")
chart_hip_gro_y.plot(hip_gro_y_vaxis, "yellowgreen", label="y_ankle_gro")
chart_hip_gro_z.plot(hip_gro_z_vaxis, "deepskyblue", label="z_ankle_gro")

chart_ankle_acc_x.set_xticks([])
chart_ankle_acc_x.set_yticks([])
chart_ankle_acc_y.set_xticks([])
chart_ankle_acc_y.set_yticks([])
chart_ankle_acc_z.set_xticks([])
chart_ankle_acc_z.set_yticks([])
chart_ankle_gro_x.set_xticks([])
chart_ankle_gro_x.set_yticks([])
chart_ankle_gro_y.set_xticks([])
chart_ankle_gro_y.set_yticks([])
chart_ankle_gro_z.set_xticks([])
chart_ankle_gro_z.set_yticks([])

chart_knee_acc_x.set_xticks([])
chart_knee_acc_x.set_yticks([])
chart_knee_acc_y.set_xticks([])
chart_knee_acc_y.set_yticks([])
chart_knee_acc_z.set_xticks([])
chart_knee_acc_z.set_yticks([])
chart_knee_gro_x.set_xticks([])
chart_knee_gro_x.set_yticks([])
chart_knee_gro_y.set_xticks([])
chart_knee_gro_y.set_yticks([])
chart_knee_gro_z.set_xticks([])
chart_knee_gro_z.set_yticks([])

chart_hip_acc_x.set_xticks([])
chart_hip_acc_x.set_yticks([])
chart_hip_acc_y.set_xticks([])
chart_hip_acc_y.set_yticks([])
chart_hip_acc_z.set_xticks([])
chart_hip_acc_z.set_yticks([])
chart_hip_gro_x.set_xticks([])
chart_hip_gro_x.set_yticks([])
chart_hip_gro_y.set_xticks([])
chart_hip_gro_y.set_yticks([])
chart_hip_gro_z.set_xticks([])
chart_hip_gro_z.set_yticks([])


# chart_ankle_acc_x.xlabel('time series')
# chart_ankle_acc_x.ylabel('value')
# chart_ankle_acc_y.xlabel('time series')
# chart_ankle_acc_y.ylabel('value')
# chart_ankle_acc_z.xlabel('time series')
# chart_ankle_acc_z.ylabel('value')
# chart_ankle_gro_x.xlabel('time series')
# chart_ankle_gro_x.ylabel('value')
# chart_ankle_gro_y.xlabel('time series')
# chart_ankle_gro_y.ylabel('value')
# chart_ankle_gro_z.xlabel('time series')
# chart_ankle_gro_z.ylabel('value')
#
# chart_knee_acc_x.xlabel('time series')
# chart_knee_acc_x.ylabel('value')
# chart_knee_acc_y.xlabel('time series')
# chart_knee_acc_y.ylabel('value')
# chart_knee_acc_z.xlabel('time series')
# chart_knee_acc_z.ylabel('value')
# chart_knee_gro_x.xlabel('time series')
# chart_knee_gro_x.ylabel('value')
# chart_knee_gro_y.xlabel('time series')
# chart_knee_gro_y.ylabel('value')
# chart_knee_gro_z.xlabel('time series')
# chart_knee_gro_z.ylabel('value')
#
# chart_hip_acc_x.xlabel('time series')
# chart_hip_acc_x.ylabel('value')
# chart_hip_acc_y.xlabel('time series')
# chart_hip_acc_y.ylabel('value')
# chart_hip_acc_z.xlabel('time series')
# chart_hip_acc_z.ylabel('value')
# chart_hip_gro_x.xlabel('time series')
# chart_hip_gro_x.ylabel('value')
# chart_hip_gro_y.xlabel('time series')
# chart_hip_gro_y.ylabel('value')
# chart_hip_gro_z.xlabel('time series')
# chart_hip_gro_z.ylabel('value')

chart_ankle_acc_x.legend()
chart_ankle_acc_y.legend()
chart_ankle_acc_z.legend()
chart_ankle_gro_x.legend()
chart_ankle_gro_y.legend()
chart_ankle_gro_z.legend()
chart_knee_acc_x.legend()
chart_knee_acc_y.legend()
chart_knee_acc_z.legend()
chart_knee_gro_x.legend()
chart_knee_gro_y.legend()
chart_knee_gro_z.legend()
chart_hip_acc_x.legend()
chart_hip_acc_y.legend()
chart_hip_acc_z.legend()
chart_hip_gro_x.legend()
chart_hip_gro_y.legend()
chart_hip_gro_z.legend()
plt.show()
