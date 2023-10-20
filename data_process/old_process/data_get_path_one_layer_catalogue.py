import os

import pandas


def get_file_name(in_root, out_root):
    in_dirs = os.listdir(in_root)

    # one layer catalogue
    out_dirs, i = [], 0
    for dir in in_dirs:
        in_dirs[i] = in_root + '/' + dir
        name = dir.split('_pr')[0]
        path_f = out_root + '/' + name
        if not os.path.exists(path_f):
            os.mkdir(path_f)
        out_dirs.insert(i, path_f)
        i += 1

    return in_dirs, out_dirs