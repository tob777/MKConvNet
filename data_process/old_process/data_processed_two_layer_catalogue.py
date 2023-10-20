import os
import data_label

def file_processed(in_root, out_root):
    in_dirs = os.listdir(in_root)
    key = 1

    # two layer catalogue
    for dir in in_dirs:
        in_path_1 = in_root + '/' + dir
        out_path_1 = out_root + '/' + dir
        if not os.path.exists(out_path_1):
            os.mkdir(out_path_1)

        in_dirs1 = os.listdir(in_path_1)
        out_dirs, i = [], 0
        for dir1 in in_dirs1:
            in_path_2 = in_path_1 + '/' + dir1
            in_dirs1[i] = in_path_2
            out_path_2 = out_path_1 + '/' + dir1
            if not os.path.exists(out_path_2):
                os.mkdir(out_path_2)
            out_dirs.insert(i, out_path_2)
            i += 1
        #label
        # for j in range(len(in_dirs1)):
        #     data_label.data_processed(in_dirs1[j], out_dirs[j], os.listdir(in_dirs1[j]))

    # rename
        for j in range(len(in_dirs1)):
            inpath = in_dirs1[j]
            files = os.listdir(inpath)
            for file in files:
                old_name = inpath + '/' + file
                new_name = inpath + '/' + str(key)+ '_' + file
                os.rename(old_name, new_name)
                key += 1

    # get_data_numbers
    # shape_list = []
    # shape_list.append()
    # shape_list = np.array(shape_list)
    # shape_list = shape_list[:, :, 0]
    # np.savetxt('D:/data_count.csv', np.transpose(shape_list), delimiter=',')

in_root = 'D:/1.data/14_train_sets_unemg_label'
out_root = 'D:/1.data/14_train_sets_unemg_label'
file_processed(in_root, out_root)