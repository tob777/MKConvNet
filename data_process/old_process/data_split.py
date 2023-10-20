import pandas as pd
import os

def data_process(file_path,files,file_processed_path):
    for file in files:
        path = file_path + '/' + file
        print(path)
        f = open(path)
        data = pd.read_csv(f, encoding='utf-8')
        data_long = pd.DataFrame()
        data_short = pd.DataFrame()
        len_short = len(data['Avanti sensor 3: ACC.X 3 [g]'].dropna(axis=0))
        len_long = len(data['X [s]'])

        for col in data.columns:
            if len(data[col].dropna(axis=0)) > len_short:
                data_long[col] = data[col]
            else:
                data_short[col] = data[col]
        data_short = data_short.iloc[296:len_short - 295, :]
        data_long = data_long.iloc[2513:len_long - 2513, :]

        file_processed_path_emg = file_processed_path + '/emg'
        if not os.path.exists(file_processed_path_emg):
            os.mkdir(file_processed_path_emg)
        file_processed_path_unemg = file_processed_path + '/unemg'
        if not os.path.exists(file_processed_path_unemg):
            os.mkdir(file_processed_path_unemg)

        path_emg = file_processed_path_emg + '/emg' + file
        path_unemg = file_processed_path_unemg + '/umemg' + file
        data_long.to_csv(path_emg, index=False)
        data_short.to_csv(path_unemg, index=False)

def get_file_name(root, root_pd):
    list_dirs = os.listdir(root)

    root1, root2 = root + '/', root_pd + '/'
    list_dirs_processed, i = [], 0
    for dir in list_dirs:
        list_dirs[i] = root1 + dir
        path_processed = root2 + dir + '_processed'
        if not os.path.exists(path_processed):
            os.mkdir(path_processed)
        list_dirs_processed.insert(i, path_processed)
        i += 1

    for i in range(len(list_dirs)):
        files = os.listdir(list_dirs[i])
        data_process(list_dirs[i], files, list_dirs_processed[i])

root = 'D:/2.data/1_data'
root_pd = 'D:/2.data/2_data_split'
get_file_name(root, root_pd)