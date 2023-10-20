import pandas as pd

def data_processed(in_dir, out_dir, files):
    shape_list = []
    for file in files:
        path = in_dir + '/' + file
        data = pd.read_csv(path, encoding = 'utf-8')
        return shape_list