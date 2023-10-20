import os
import pandas as pd

in_root = 'D:/3.data/1.raw_data/3.twelve'
out_root = 'D:/3.data/2.data_unemg_lowb/3.twelve'

in_files_list = os.listdir(in_root)

for subject_name in in_files_list:
    subject_path = os.path.join(in_root, subject_name)
    subject = os.listdir(subject_path)
    for file in subject:
        print(subject_name, ' ', file)
        f = open(os.path.join(subject_path, file))
        data = pd.read_csv(f)

        cols = [
            'Avanti sensor 3: ACC.X 3 [g]', 'Avanti sensor 3: ACC.Y 3 [g]', 'Avanti sensor 3: ACC.Z 3 [g]',
            'Avanti sensor 3: GYRO.X 3 [°/s]', 'Avanti sensor 3: GYRO.Y 3 [°/s]', 'Avanti sensor 3: GYRO.Z 3 [°/s]',
            'Avanti sensor 5: ACC.X 5 [g]', 'Avanti sensor 5: ACC.Y 5 [g]', 'Avanti sensor 5: ACC.Z 5 [g]',
            'Avanti sensor 5: GYRO.X 5 [°/s]', 'Avanti sensor 5: GYRO.Y 5 [°/s]', 'Avanti sensor 5: GYRO.Z 5 [°/s]',
            'Avanti sensor 7: ACC.X 7 [g]', 'Avanti sensor 7: ACC.Y 7 [g]', 'Avanti sensor 7: ACC.Z 7 [g]',
            'Avanti sensor 7: GYRO.X 7 [°/s]', 'Avanti sensor 7: GYRO.Y 7 [°/s]', 'Avanti sensor 7: GYRO.Z 7 [°/s]',
            'Avanti sensor 8: ACC.X 8 [g]', 'Avanti sensor 8: ACC.Y 8 [g]', 'Avanti sensor 8: ACC.Z 8 [g]',
            'Avanti sensor 8: GYRO.X 8 [°/s]', 'Avanti sensor 8: GYRO.Y 8 [°/s]', 'Avanti sensor 8: GYRO.Z 8 [°/s]',
            'Avanti sensor 6: ACC.X 6 [g]', 'Avanti sensor 6: ACC.Y 6 [g]', 'Avanti sensor 6: ACC.Z 6 [g]',
            'Avanti sensor 6: GYRO.X 6 [°/s]', 'Avanti sensor 6: GYRO.Y 6 [°/s]', 'Avanti sensor 6: GYRO.Z 6 [°/s]',
            'Avanti sensor 4: ACC.X 4 [g]', 'Avanti sensor 4: ACC.Y 4 [g]', 'Avanti sensor 4: ACC.Z 4 [g]',
            'Avanti sensor 4: GYRO.X 4 [°/s]', 'Avanti sensor 4: GYRO.Y 4 [°/s]', 'Avanti sensor 4: GYRO.Z 4 [°/s]',
        ]


        data = data[cols]
        data = data[296:len(data.dropna(axis=0)) - 295]
        print(data.shape)
        data.drop_duplicates(subset=None, keep=False, inplace=True)
        print(data.shape)
        out_dic = os.path.join(out_root, subject_name)
        os.makedirs(out_dic, exist_ok=True)

        data.to_csv(os.path.join(out_dic, file), index=False)