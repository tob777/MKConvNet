import sys

import pyKinectAzure.pykinect_azure as pykinect
import pyKinectAzure.pykinect_azure.k4abt._k4abtTypes as btTypes
import os.path
import cv2
import pandas as pd
sys.path.insert(1, '../')


def get_body_joints(bodies, dic):
    for i in range(len(bodies)):
        skeleton = bodies[i].joints
        data = []
        for j in range(btTypes.K4ABT_JOINT_COUNT):  # 32个关节点 * 3个坐标 * 4个四元数坐标 = 224
            position = skeleton[j].position
            orientation = skeleton[j].orientation
            data.append(position.x)
            data.append(position.y)
            data.append(position.z)

            data.append(orientation.w)
            data.append(orientation.x)
            data.append(orientation.y)
            data.append(orientation.z)
        data = pd.Series(data)
        dic[i] = dic[i].append(data, ignore_index=True)

    return dic


# confidence_level = skeleton[j].confidence_level
# with open('../../../data/body_joints.csv', 'a') as f:
# 	f.write(','.join([str(x) + ' ' + str(y) + ' ' + str(z) for x, y, z in position]) + '\n')

if __name__ == "__main__":
    # 保存文件路径
    filepath = 'D:\\atest'
    df_dict = {0: pd.DataFrame(), 1: pd.DataFrame(),
               2: pd.DataFrame(), 3: pd.DataFrame(),
               4: pd.DataFrame()}

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries(track_body=True, module_k4a_path=None, module_k4abt_path=None)

    # Modify camera configuration
    device_config = pykinect.default_configuration
    # device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_OFF
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_UNBINNED
    # print(device_config)

    # Start device
    device = pykinect.start_device(config=device_config)

    # Start body tracker
    bodyTracker = pykinect.start_body_tracker()

    cv2.namedWindow('Depth image with skeleton', cv2.WINDOW_NORMAL)
    while True:

        # Get capture
        capture = device.update()

        # Get body tracker frame
        body_frame = bodyTracker.update()
        # print(body_frame)
        # Get the color depth image from the capture
        ret, depth_color_image = capture.get_colored_depth_image()

        # Get the colored body segmentation
        ret, body_image_color = body_frame.get_segmentation_image()

        if not ret:
            continue

        # Combine both images
        combined_image = cv2.addWeighted(depth_color_image, 0.6, body_image_color, 0.4, 0)
        # 获取所有捕捉到的人体，至多捕捉5人
        bodies = body_frame.get_bodies()
        df_dict = get_body_joints(bodies, df_dict)

        # Draw the skeletons
        combined_image = body_frame.draw_bodies(combined_image)
        # test = body_frame.get_body_skeleton()
        # Overlay body segmentation on depth image
        cv2.imshow('Depth image with skeleton', combined_image)

        # Press q key to stop
        if cv2.waitKey(1) == ord('q'):
            # 按q结束采集，准备保存数据
            for i in range(5):
                if len(df_dict[i]) == 0:
                    continue
                else:
                    df_dict[i].to_csv(os.path.join(filepath, str(i) + '.csv'))
                    # 保存文件成功
            break
