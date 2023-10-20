import os
import data_get_path_one_layer_catalogue
import get_unemg_all

def file_processed(in_root, out_root):
    in_dirs, out_dirs = data_get_path_one_layer_catalogue.get_file_name(in_root, out_root)

    for i in range(len(in_dirs)):
        get_unemg_all.data_processed(in_dirs[i], out_dirs[i], os.listdir(in_dirs[i]))

    # get_data_numbers
    # shape_list = []
    # shape_list.append()
    # shape_list = np.array(shape_list)
    # shape_list = shape_list[:, :, 0]
    # np.savetxt('D:/data_count.csv', np.transpose(shape_list), delimiter=',')

in_root = 'D:/1.data/3_data_unemg_uf'
out_root = 'D:/1.data/16_unemg_uf'
file_processed(in_root, out_root)