import pandas as pd
import os

def data_processed(in_root, out_root, files):
    for file in files:
        data = pd.read_csv(in_root + '/' + file)

        i, tip, cnt = 0, 3, 1
        for col in data.columns:
            # un_emg
            if i % 12 == 0:
                data_pd = pd.DataFrame()
                out_root_file = out_root + '/s' + str(tip)
                if not os.path.exists(out_root_file):
                    os.mkdir(out_root_file)
                tip += 1
            if i % 2 != 0:
                data_pd[col] = data[col]
            if cnt % 12 == 0:
                data_pd.to_csv(out_root_file + '/' + 's' + str(tip - 1) + '_' + file, index=False)
            i += 1
            cnt += 1

            # emg
            # if i % 2 != 0:
            #     data_pd = pd.DataFrame()
            #     out_root_file = out_root + '/s' + str(tip)
            #     if not os.path.exists(out_root_file):
            #         os.mkdir(out_root_file)
            #     data_pd[col] = data[col]
            #     data_pd.to_csv(out_root_file + '/' + 's' + str(tip) + '_' + file, index=False)
            #     tip += 1
            # i += 1

