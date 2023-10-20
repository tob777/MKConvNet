import os

import pandas as pd

def data_processed(in_root, out_root, files):
    print(files)
    for file in files:
        data = pd.read_csv(in_root + '/' + file)
        print(in_root, file)
        cols1 = [
            'Avanti sensor 3: ACC.X 3 [g]', 'Avanti sensor 3: ACC.Y 3 [g]', 'Avanti sensor 3: ACC.Z 3 [g]',
            'Avanti sensor 3: GYRO.X 3 [°/s]', 'Avanti sensor 3: GYRO.Y 3 [°/s]', 'Avanti sensor 3: GYRO.Z 3 [°/s]',
            'Avanti sensor 5: ACC.X 5 [g]',	'Avanti sensor 5: ACC.Y 5 [g]',	'Avanti sensor 5: ACC.Z 5 [g]',
            'Avanti sensor 5: GYRO.X 5 [°/s]', 'Avanti sensor 5: GYRO.Y 5 [°/s]', 'Avanti sensor 5: GYRO.Z 5 [°/s]',
            'Avanti sensor 7: ACC.X 7 [g]', 'Avanti sensor 7: ACC.Y 7 [g]', 'Avanti sensor 7: ACC.Z 7 [g]',
            'Avanti sensor 7: GYRO.X 7 [°/s]', 'Avanti sensor 7: GYRO.Y 7 [°/s]', 'Avanti sensor 7: GYRO.Z 7 [°/s]',

        ]
        cols2 = [
            'Avanti sensor 8: ACC.X 8 [g]', 'Avanti sensor 8: ACC.Y 8 [g]', 'Avanti sensor 8: ACC.Z 8 [g]',
            'Avanti sensor 8: GYRO.X 8 [°/s]', 'Avanti sensor 8: GYRO.Y 8 [°/s]', 'Avanti sensor 8: GYRO.Z 8 [°/s]',
            'Avanti sensor 6: ACC.X 6 [g]', 'Avanti sensor 6: ACC.Y 6 [g]', 'Avanti sensor 6: ACC.Z 6 [g]',
            'Avanti sensor 6: GYRO.X 6 [°/s]', 'Avanti sensor 6: GYRO.Y 6 [°/s]', 'Avanti sensor 6: GYRO.Z 6 [°/s]',
            'Avanti sensor 4: ACC.X 4 [g]', 'Avanti sensor 4: ACC.Y 4 [g]', 'Avanti sensor 4: ACC.Z 4 [g]',
            'Avanti sensor 4: GYRO.X 4 [°/s]', 'Avanti sensor 4: GYRO.Y 4 [°/s]', 'Avanti sensor 4: GYRO.Z 4 [°/s]',
        ]
        data1 = data[cols1]
        data2 = data[cols2]
        data = pd.concat((data1, data2), axis=1)

        name = file[5:]
        data.to_csv(out_root + '/' + name, index=False)
