from scipy import signal
import pandas as pd
import numpy as np

def filter_select(N, Wn, tensor, filter_type):
    b, a = signal.iirfilter(N, Wn, btype='bandstop', ftype=filter_type)
    output = signal.filtfilt(b, a, tensor, axis=0)
    return output

def data_processed(in_dir, out_dir, files):
    for file in files:
        path = in_dir + '/' + file
        data = pd.read_csv(path, encoding = 'utf-8')
        data_filtered = []

        cnt1, cnt2 = 0, 0
        t1, t2 = 0, 1
        while cnt1 < data.shape[1]:
            filtered = filter_select(8, [0.06, 0.995], data.iloc[:, cnt1: cnt1 + 2], 'butter')
            data_filtered.insert(cnt2, filtered[:, t1])
            cnt2 += 1
            data_filtered.insert(cnt2, filtered[:, t2])
            cnt2 += 1
            cnt1 += 2
        data_filtered = pd.DataFrame(np.array(data_filtered).transpose(), columns=data.columns)
        data_filtered.to_csv(out_dir + '/f_' + file, index=False)