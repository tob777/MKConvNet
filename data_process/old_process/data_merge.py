import os
import pandas as pd

def file_processed(in_root_unemg, in_root_emg, out_root):
    in_dirs_ue = os.listdir(in_root_unemg)
    in_dirs_e = os.listdir(in_root_emg)

    # two layer catalogue
    for i in range(len(in_dirs_e)):
        in_path_ue1 = in_root_unemg + '/' + in_dirs_ue[i]
        in_path_e1 = in_root_emg + '/' + in_dirs_e[i]
        out_path_1 = out_root + '/' + in_dirs_e[i]
        if not os.path.exists(out_path_1):
            os.mkdir(out_path_1)

        in_dirs_ue1 = os.listdir(in_path_ue1)
        in_dirs_e1 = os.listdir(in_path_e1)
        in_dirs_unemg, in_dirs_emg, out_dirs = [], [], []
        for j in range(len(in_dirs_e1)):
            in_path_ue2 = in_path_ue1 + '/' + in_dirs_ue1[j]
            in_path_e2 = in_path_e1 + '/' + in_dirs_e1[j]
            in_dirs_unemg.insert(j, in_path_ue2)
            in_dirs_emg.insert(j, in_path_e2)

            out_path_2 = out_path_1 + '/' + in_dirs_e1[j]
            if not os.path.exists(out_path_2):
                os.mkdir(out_path_2)
            out_dirs.insert(j, out_path_2)

        for k in range(len(in_dirs_emg)):
            data_processed(in_dirs_unemg[k], in_dirs_emg[k], out_dirs[k], os.listdir(in_dirs_unemg[k]), os.listdir(in_dirs_emg[k]))

    # get_data_numbers
    # shape_list = []
    # shape_list.append()
    # shape_list = np.array(shape_list)
    # shape_list = shape_list[:, :, 0]
    # np.savetxt('D:/data_count.csv', np.transpose(shape_list), delimiter=',')

def data_processed(in_root_ue, in_root_e, out_root, files_ue, files_e):
    for tip in range(len(files_e)):
        file_ue = in_root_ue + '/' + files_ue[tip]
        file_e = in_root_e + '/' + files_e[tip]
        data_ue = pd.read_csv(file_ue)
        data_e = pd.read_csv(file_e)
        print(file_e, ' ', file_ue)

        data = pd.concat([data_ue, data_e], axis=1)
        data.dropna(axis=0, inplace=True)

        file_name = files_e[tip][5:len(files_e[tip])]
        out_root_file = out_root + '/' + file_name
        print(out_root_file)
        data.to_csv(out_root_file, index=False)

in_root_unemg = 'D:/1.data/5_data_sets_unemg'
in_root_emg = 'D:/1.data/8_data_sets_emg'
out_root = 'D:/1.data/9_data_merge'
file_processed(in_root_unemg, in_root_emg, out_root)