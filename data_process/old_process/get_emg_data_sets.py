import numpy as np
import pandas as pd

def data_processed(in_root, out_root, files):
    for file in files:
        data = np.loadtxt(in_root + '/' + file, delimiter=',', skiprows=1)
        data.transpose()

        row = len(data) // 8
        col = 8
        data = data[np.arange(0, col * row)].reshape(row, col)
        data_emg = pd.DataFrame(data, columns=['emg1', 'emg2', 'emg3', 'emg4', 'emg5', 'emg6', 'emg7', 'emg8'])

        out_root_file = out_root + '/' + file
        data_emg.to_csv(out_root_file, index=False)