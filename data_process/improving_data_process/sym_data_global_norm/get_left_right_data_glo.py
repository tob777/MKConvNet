import os
import pandas as pd

in_root = 'D:/3.data/1.raw_data/4.nine'
out_root = 'D:/3.data/2.data_unemg_lowb/6.data_sym_global_norm_9/3.data_sym_glo'

in_files_list = os.listdir(in_root)
cols_left = [3, 5, 7, 9, 11, 13, 31, 33, 35, 37, 39, 41, 59, 61, 63, 65, 67, 69]
cols_right = [17, 19, 21, 23, 25, 27, 45, 47, 49, 51, 53, 55, 73, 75, 77, 79, 81, 83]

for subject_name in in_files_list:
    subject_path = os.path.join(in_root, subject_name)
    subject = os.listdir(subject_path)
    for file in subject:
        print(subject_name, ' ', file)
        f = open(os.path.join(subject_path, file))
        data = pd.read_csv(f)

        # cols_left = [
        #     'Avanti sensor 3: ACC.X 3 [g]', 'Avanti sensor 3: ACC.Y 3 [g]', 'Avanti sensor 3: ACC.Z 3 [g]',
        #     'Avanti sensor 3: GYRO.X 3 [～/s]', 'Avanti sensor 3: GYRO.Y 3 [～/s]', 'Avanti sensor 3: GYRO.Z 3 [～/s]',
        #     'Avanti sensor 5: ACC.X 5 [g]', 'Avanti sensor 5: ACC.Y 5 [g]', 'Avanti sensor 5: ACC.Z 5 [g]',
        #     'Avanti sensor 5: GYRO.X 5 [～/s]', 'Avanti sensor 5: GYRO.Y 5 [～/s]', 'Avanti sensor 5: GYRO.Z 5 [～/s]',
        #     'Avanti sensor 7: ACC.X 7 [g]', 'Avanti sensor 7: ACC.Y 7 [g]', 'Avanti sensor 7: ACC.Z 7 [g]',
        #     'Avanti sensor 7: GYRO.X 7 [～/s]', 'Avanti sensor 7: GYRO.Y 7 [～/s]', 'Avanti sensor 7: GYRO.Z 7 [～/s]',
        # ]
        #
        # cols_right = [
        #     'Avanti sensor 4: ACC.X 4 [g]', 'Avanti sensor 4: ACC.Y 4 [g]', 'Avanti sensor 4: ACC.Z 4 [g]',
        #     'Avanti sensor 4: GYRO.X 4 [～/s]', 'Avanti sensor 4: GYRO.Y 4 [～/s]', 'Avanti sensor 4: GYRO.Z 4 [～/s]',
        #     'Avanti sensor 6: ACC.X 6 [g]', 'Avanti sensor 6: ACC.Y 6 [g]', 'Avanti sensor 6: ACC.Z 6 [g]',
        #     'Avanti sensor 6: GYRO.X 6 [～/s]', 'Avanti sensor 6: GYRO.Y 6 [～/s]', 'Avanti sensor 6: GYRO.Z 6 [～/s]',
        #     'Avanti sensor 8: ACC.X 8 [g]', 'Avanti sensor 8: ACC.Y 8 [g]', 'Avanti sensor 8: ACC.Z 8 [g]',
        #     'Avanti sensor 8: GYRO.X 8 [～/s]', 'Avanti sensor 8: GYRO.Y 8 [～/s]', 'Avanti sensor 8: GYRO.Z 8 [～/s]',
        # ]

        # cols_left = [
        #     'Avanti sensor 3: ACC.X 3 [g]', 'Avanti sensor 3: ACC.Y 3 [g]', 'Avanti sensor 3: ACC.Z 3 [g]',
        #     'Avanti sensor 3: GYRO.X 3 [°/s]', 'Avanti sensor 3: GYRO.Y 3 [°/s]', 'Avanti sensor 3: GYRO.Z 3 [°/s]',
        #     'Avanti sensor 5: ACC.X 5 [g]', 'Avanti sensor 5: ACC.Y 5 [g]', 'Avanti sensor 5: ACC.Z 5 [g]',
        #     'Avanti sensor 5: GYRO.X 5 [°/s]', 'Avanti sensor 5: GYRO.Y 5 [°/s]', 'Avanti sensor 5: GYRO.Z 5 [°/s]',
        #     'Avanti sensor 7: ACC.X 7 [g]', 'Avanti sensor 7: ACC.Y 7 [g]', 'Avanti sensor 7: ACC.Z 7 [g]',
        #     'Avanti sensor 7: GYRO.X 7 [°/s]', 'Avanti sensor 7: GYRO.Y 7 [°/s]', 'Avanti sensor 7: GYRO.Z 7 [°/s]',
        # ]
        #
        # cols_right = [
        #     'Avanti sensor 4: ACC.X 4 [g]', 'Avanti sensor 4: ACC.Y 4 [g]', 'Avanti sensor 4: ACC.Z 4 [g]',
        #     'Avanti sensor 4: GYRO.X 4 [°/s]', 'Avanti sensor 4: GYRO.Y 4 [°/s]', 'Avanti sensor 4: GYRO.Z 4 [°/s]',
        #     'Avanti sensor 6: ACC.X 6 [g]', 'Avanti sensor 6: ACC.Y 6 [g]', 'Avanti sensor 6: ACC.Z 6 [g]',
        #     'Avanti sensor 6: GYRO.X 6 [°/s]', 'Avanti sensor 6: GYRO.Y 6 [°/s]', 'Avanti sensor 6: GYRO.Z 6 [°/s]',
        #     'Avanti sensor 8: ACC.X 8 [g]', 'Avanti sensor 8: ACC.Y 8 [g]', 'Avanti sensor 8: ACC.Z 8 [g]',
        #     'Avanti sensor 8: GYRO.X 8 [°/s]', 'Avanti sensor 8: GYRO.Y 8 [°/s]', 'Avanti sensor 8: GYRO.Z 8 [°/s]',
        # ]

        cols = data.columns

        data_left = data[cols[cols_left]]
        data_left = data_left[296:len(data.dropna(axis=0)) - 295]
        data_right = data[cols[cols_right]]
        data_right = data_right[296:len(data.dropna(axis=0)) - 295]

        print(data_left.shape, ' ', data_right.shape)
        data_left.drop_duplicates(subset=None, keep=False, inplace=True)
        data_right.drop_duplicates(subset=None, keep=False, inplace=True)
        print(data_left.shape, ' ', data_right.shape)

        # normalization
        # left_col = data_left.columns
        # data_left = data_left[left_col].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        # right_col = data_right.columns
        # data_right = data_right[right_col].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        out_root_left = out_root + '/' + 'left_body'
        out_root_right = out_root + '/' + 'right_body'
        out_dic_left = os.path.join(out_root_left, subject_name)
        out_dic_right = os.path.join(out_root_right, subject_name)
        os.makedirs(out_dic_left, exist_ok=True)
        os.makedirs(out_dic_right, exist_ok=True)

        data_left.to_csv(os.path.join(out_dic_left, file), index=False)
        data_right.to_csv(os.path.join(out_dic_right, file), index=False)