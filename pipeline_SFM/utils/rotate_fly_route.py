
# -*- coding: utf-8 -*-
# import piexif
from PIL import Image
import os
import sys
# import pyexiv2  
import numpy as np
import cv2
from PIL import Image
from pyproj import Transformer
from fractions import Fraction
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# pyexiv2.set_log_level(4)
#使用pyexiv2 2.4.1版本
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
print(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__),"pipeline_SFM"))
sys.path.append(os.path.join(os.path.dirname(__file__),"pipeline_SFM", "utils"))

from pipeline_SFM.utils.read_exif_data import parse_GPS_info_to_xyz

'''
export PYTHONPATH="/home/xavision/miniconda3/envs/xfeat/lib/python3.8/site-packages:$PYTHONPATH"
'''

def DetectFlightRoute(x, y):
    if len(x) != len(y):
        return []
    if len(x) <= 2:
        return []

    mbSaveLastRouteLine = False
    mFlightRoute = -1
    _data_set_size = len(x)
    _cur_flight_direction = []
    _former_flight_direction = []
    _former_frame_id = []
    _flight_route = []
    print("data set size: ", _data_set_size)
    for idx in range(0, _data_set_size - 2):
        if mbSaveLastRouteLine == False:
            mbSaveLastRouteLine = True
            mFlightRoute = 1

        _diff_x1 = x[idx] - x[idx + 1]
        _diff_y1 = y[idx] - y[idx + 1]
        _sqrtD1 = 1.0 / np.sqrt(_diff_x1 * _diff_x1 + _diff_y1 * _diff_y1)

        _diff_x2 = x[idx + 1] - x[idx + 2]
        _diff_y2 = y[idx + 1] - y[idx + 2]

        if len(_cur_flight_direction) != 0:
            _diff_x2 = _cur_flight_direction[0]
            _diff_y2 = _cur_flight_direction[1]
        
        _sqrtD2 = 1.0 / np.sqrt(_diff_x2 * _diff_x2 + _diff_y2 * _diff_y2)
        cosValue = (_diff_x1 * _diff_x2 + _diff_y1 * _diff_y2) * _sqrtD1 * _sqrtD2
        print(" cur frame idx: ", idx, " cosValue ", cosValue)

        if cosValue > 0.95:
            if len(_cur_flight_direction) == 0:
                _cur_flight_direction.append(x[idx] - x[idx + 1])
                _cur_flight_direction.append(y[idx] - y[idx + 1])
                _former_flight_direction = _cur_flight_direction
                _former_frame_id.append(idx)
                _former_frame_id.append(idx + 1)
                _former_frame_id.append(idx + 2)
            print("_cur_flight_direction: ", _cur_flight_direction[0], " ", _cur_flight_direction[1])
        elif cosValue < -0.95:
            if len(_cur_flight_direction) != 0:
                mFlightRoute = mFlightRoute + 1
                _cur_flight_direction[0] = x[idx] - x[idx + 1]
                _cur_flight_direction[1] = y[idx] - y[idx + 1]
                print("another_flight_direction: ", _cur_flight_direction[0], " ", _cur_flight_direction[1])
            print("mFlightRoute ", mFlightRoute)

        _flight_route.append(mFlightRoute)
    ## Last 2 Frame using same flight route
    _flight_route.append(mFlightRoute)
    _flight_route.append(mFlightRoute)
    return _flight_route



def DrawECEFPosition(x, y, z):
    fig = plt.figure()
    #创建3d绘图区域
    ax = plt.axes(projection='3d')
    #调用 ax.plot3D创建三维线图
    # x = np.array(x)
    # y = np.array(y)
    vals = []
    for i in range(len(x)):
        vals.append([x[i], y[i]])
    vals = np.array(vals)
    # z = np.array(z)
    # ax.plot3D(x, y, z, 'gray')
    # ax.scatter3D(x, y, z, color = "red")
    # plt.scatter(x, y)  
    plt.plot(vals)
    # start position
    # ax.scatter3D(x[0], y[0], z[0], color = "blue")
    # ax.set_title('3D line plot')
    # plt.show()
    plt.savefig("gps_pose.png")





def TransformUTM(lat = 23.336855556, lon = 113.483155556, att = 0.0):
    # 参数1：WGS84地理坐标系统 对应 4326
    # 参数2：坐标系WKID 广州市 WGS_1984_UTM_Zone_49N 对应 32649
    transformer = Transformer.from_crs("epsg:4326", "epsg:32649")
    x, y, z = transformer.transform(lat, lon, att)
    return [x, y, z]

def TransformWGS84(x, y, z):
    # 参数1：WGS84地理坐标系统 对应 4326
    # 参数2：坐标系WKID 广州市 WGS_1984_UTM_Zone_49N 对应 32649
    # print("TransformWGS84 x: ", x, " y: ", y, " z: ", z)
    transformer = Transformer.from_crs("epsg:32649", "epsg:4326")
    lat, lon, att = transformer.transform(x, y, z)
    print("TransformWGS84 lat: ", lat, " lon: ", lon, " att: ", att)
    [tx, ty, tz] = TransformUTM(lat, lon, att)
    # print("TransformWGS84 tx: ", tx, " ty: ", ty, " tz: ", tz)
    imgCoor = np.array([x, y, z])
    target = np.array([tx, ty, tz])
    diff = imgCoor - target
    print("TransformWGS84 diff: ", np.sqrt(np.dot(diff, diff)))
    return [lat, lon, att]

#计算第一帧图像与每一帧图像之间的yaw偏差值
def CalculateYawDif(yaw_1, yaw_i):
    yaw_1_value = yaw_1.split("/")
    yaw_1 = float(yaw_1_value[0]) / float(yaw_1_value[1])
    yaw_i_value = yaw_i.split("/")
    yaw_i = float(yaw_i_value[0]) / float(yaw_i_value[1])
    yaw_dif = yaw_1 - yaw_i
    return yaw_dif

def rotate_img(inputFile, outputFile):
    
    assert os.path.isfile(inputFile)
    imgMat = cv2.imread(inputFile)
    imgMat = np.rot90(imgMat, 2)
    cv2.imwrite(outputFile, imgMat)
    
    return 


if __name__ == "__main__":
    # if len(sys.argv) < 3:
    #     print ("Usage %s file_input file_output" % sys.argv[0])
    #     #   file_input:   架次数据的文件夹
    #     #   file_output:  架次输出的文件夹
    #     sys.exit(1)

    file_input = "/home/xavision/nnd_storage_0/Asher/data/PV_wheat_field/29e03514870de0e379b36381"
    file_output = "/home/xavision/nnd_storage_0/Asher/data/PV_wheat_field/29e03514870de0e379b36381/rotate"
    files = os.listdir(file_input)
    files.sort()
    files_nums=len(files)
    _data_set_type = "NONE"
    _base_x = -1
    _base_y = -1
    _base_z = -1
    _ecef_x = []
    _ecef_y = []
    _ecef_z = []
    _image_lists = []
    j = 0
    for i in range(files_nums):
        if not files[i].endswith('.JPG'):
            j = i
        else:
            first_img_index = j + 1
            file_path_1 = file_input + "/" + files[first_img_index]
            file_input_path = file_input + "/" + files[i]   # 输入文件名
            file_output_path = file_output + "/" + files[i]  # 输出文件名
            # check data set type
            if _data_set_type == "NONE":
                image = Image.open(file_input_path)
                print("img_width : ", image.size[0])
                print("img_height : ", image.size[1])
                if image.size[0] == 2000 and image.size[1] == 1500:
                    _data_set_type = "PV"
                else:
                    _data_set_type = "XMission"
                print("data set type : ", _data_set_type)
            
            if _data_set_type == "XMission":

                pass
            elif _data_set_type == "PV":  # 执行这一条
  
                pass
            else:
                print("Invalid Data Set")
    if _data_set_type == "PV":
        
        _image_lists = [file for file in files if file.endswith(".JPG")]
        _image_lists.sort()
        _ecef_x, _ecef_y, _ecef_z = parse_GPS_info_to_xyz(file_input, _image_lists)

    if _data_set_type == "PV":
        _flight_route = DetectFlightRoute(_ecef_x, _ecef_y)
        # DrawECEFPosition(_ecef_x, _ecef_y, _flight_route)
        for idx in range(0, len(_flight_route)):
            _fr = _flight_route[idx]
            file_input_path = file_input + "/" + _image_lists[idx]
            file_output_path = file_output + "/" + _image_lists[idx]
            print("file_output_path: ", file_output_path)
            if _fr % 2 != 0:
                print("旋转")
                rotate_img(file_input, file_output)
                # ModifyRotatePicExifXmp(file_input_path, file_output_path)
            else:
                print("不旋转")
                # ModifyNotRotatePicExifXmp(file_input_path, file_output_path)





'''
python rotate_fly_route.py /home/xavision/nnd_storage_0/Asher/data/PV_wheat_field/29e03514870de0e379b36381 /home/xavision/nnd_storage_0/Asher/data/PV_wheat_field/29e03514870de0e379b36381/rotate

'''