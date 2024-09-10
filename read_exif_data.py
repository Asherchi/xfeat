

'''
1. 从exif数据中读取GPS信息
2. GPS信息解析为局部位置 这里可以暂时不用考虑高程
3. 构建2D树，然后对每一个数据查找最近的N个点-N可以冗余一点点
4. 输出最相近的N个点，因为对每个店都会计算最近的N个点，所以肯定需要去重

'''

import os
import utm
# from PIL import Image  
# from PIL import ExifTags 
# import exifread
# piexif.load
# import piexif

from sklearn.neighbors import KDTree

def read_data_from_exif(image_path):

    exif_dict = dict()
    # exif_dict = piexif.load(image_path) 
    import json
    import subprocess
    exiftool_output = subprocess.check_output(['exiftool', '-j', image_path])

    # 解析JSON格式的输出
    metadata = json.loads(exiftool_output.decode('utf-8'))

    # metadata[0]["GPSLatitude"], metadata[0]["GPSLongitude"]

    return metadata  


def convert_GPS_data_to_shifenmiao(exif_data, gps_Str):
    '''
      gps_str: 'GPS GPSLatitude'
      gps_str: 'GPS GPSLongitude'
    '''
    GPSval = exif_data[0][gps_Str]
    # longtitude = exif_data['GPS GPSLongitude']

    latitude_shi = int(GPSval.split(" ")[0])
    latitude_fen = int(GPSval.split(" ")[2].split("'")[0])
    latitude_miao = float(GPSval.split(" ")[3].split('"')[0])

    gps_double = latitude_shi + latitude_fen / 60 + latitude_miao / 3600
    
    return gps_double


def parse_GPS_to_local_pos(latitude, lontitude, easting, northing):
    '''
        'GPS GPSLatitude'
        'GPS GPSLongitude'
    '''
    utm_first = utm.from_latlon(latitude, lontitude)
    tmpEasting, tmpNorthing = utm_first[0], utm_first[1]

    x, y = easting - tmpEasting, northing - tmpNorthing

    return (x, y)



def find_N_neareast_pos(posList, topK=5):
    
    tree = KDTree(posList)
    matchPairDict = dict()
    for idx, point in enumerate(posList):
        dists, indices = tree.query([point], k=topK)
        matchPairDict[idx] = indices[0] 

    return matchPairDict


def remove_duplicate(matchPairDict):

    pairList = list()
    for key in matchPairDict.keys():
        valsList = matchPairDict[key]
        for val in valsList:
            if val == key:
                continue
            if (key, val) in pairList or (val, key) in pairList:
                continue
            pairList.append((key, val))

    return pairList


def parse_GPS_info_to_matches_pair(imgPath, imgList, topK=7):
    
    localPos = []
    # firstFlag = False
    easting, northing, zone_number, zone_letter = None, None, None, None
    posList = []
    for idx, name in enumerate(imgList):
        imgFile = os.path.join(imgPath, name)
        assert os.path.isfile(imgFile)
        # 读取exif信息
        exif_data = read_data_from_exif(imgFile)
        # 解析为local pos 
        latitude = convert_GPS_data_to_shifenmiao(exif_data, gps_Str="GPSLatitude")
        lontitude = convert_GPS_data_to_shifenmiao(exif_data, gps_Str="GPSLongitude")

        if easting is None or northing is None:
            easting, northing, zone_number, zone_letter = utm.from_latlon(latitude, lontitude)
        
        pos = parse_GPS_to_local_pos(latitude=latitude, lontitude=lontitude, easting=easting, northing=northing)
        posList.append(pos)

    matchesPiarDict = find_N_neareast_pos(posList, topK=topK)

    filterMatchesPair = remove_duplicate(matchesPiarDict)

    return filterMatchesPair


def save_match_pair(savefile, datas):

    lines = []
    with open(savefile, "w") as f:
        for data in datas:
            saveStr = str(data[0]) + " " + str(data[1]) + "\n"
            lines.append(saveStr)
        f.writelines(lines)

    return

def run_spatial_search_matches_pair(imagesPath, saveFile, topK=7):

    # imagesPath = "/mnt/c/Users/Asher/Desktop/Data/PV_wheat_field/29e03514870de0e379b36381"
    # saveFile = "/mnt/c/Users/Asher/Desktop/Data/tmp.txt"
    nameList = os.listdir(imagesPath)
    imgList = [name for name in nameList if name.split(".")[-1] == "JPG"]
    imgList = sorted(imgList)
    print(imgList)
    filterMatchesPair = parse_GPS_info_to_matches_pair(imagesPath, imgList, topK=topK)
    save_match_pair(saveFile, filterMatchesPair)
    print("finished generate spatial search programs.")

    return filterMatchesPair


if __name__ == "__main__":
    imagesPath = "/mnt/c/Users/Asher/Desktop/Data/PV_wheat_field/29e03514870de0e379b36381"
    saveFile = "/mnt/c/Users/Asher/Desktop/Data/tmp.txt"
    run_spatial_search_matches_pair(imagesPath, saveFile=saveFile)
    # nameList = os.listdir(imagesPath)
    # imgList = [name for name in nameList if name.split(".")[-1] == "JPG"]
    # imgList = sorted(imgList)
    # print(imgList)
    # filterMatchesPair = parse_GPS_info_to_matches_pair(imagesPath, imgList)
    # save_match_pair(saveFile, filterMatchesPair)
    # print("finished generate spatial search programs.")
    pass