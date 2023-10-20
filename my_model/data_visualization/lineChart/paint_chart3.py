import matplotlib.pyplot as plt
import pandas as pd


path = "D:/3.data/1.raw_data/1.six/chen_yu_lin/下身骨科靴0度右脚1.csv"
f = open(path)
data = pd.read_csv(f)

ankle_acc_x_vaxis = data.iloc[400:1400, 3].values
x = range(len(ankle_acc_x_vaxis))

# plt.style.use('bmh')
plt.figure(figsize=(10, 3))
# plt.title("时间序列")
# plt.xlabel("时间")
# plt.xticks(rotation=45)
# plt.ylabel("指数")
plt.plot(x, ankle_acc_x_vaxis, color='r', linewidth=1)

plt.xlim(0, 1000)
# plt.ylim(-5, 6)
plt.grid(axis="both")

plt.show()
