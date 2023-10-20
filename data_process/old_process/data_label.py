import pandas as pd

def data_processed(in_root, out_root, files):
    label = [0, 0, 1, 1, 2, 2, 3, 3, 5, 5, 4, 4, 6, 6, 7, 7, 8, 8]
    for i in range(len(files)):
        file = in_root + '/' + files[i]
        data = pd.read_csv(file)
        print(file)

        data['label'] = label[i]
        data.to_csv(file, index=False)