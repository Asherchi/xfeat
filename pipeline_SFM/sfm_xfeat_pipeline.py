
import numpy as np

import torch
import os
import subprocess
import sys

import time

import cv2
import matplotlib.pyplot as plt

# import cv2
import numpy as np
import time

sys.path.append('/home/xavision/miniconda3/envs/xfeat/lib/python3.8/site-packages')
sys.path.append("/home/asher/myVersion/xfeat/modules")
sys.path.append("/home/asher/myVersion/xfeat")

from modules.xfeat import XFeat as _XFeat
# from modules.xfeat import XFeat a

'''
export PYTHONPATH="/home/xavision/miniconda3/envs/xfeat/lib/python3.8/site-packages:$PYTHONPATH"
'''

# 172.31.178.53
OPENMVG_SFM_BIN = "/home/xavision/nnd_storage_0/Asher/code/openMVG/build/Linux-x86_64-Release"
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/xavision/nnd_storage_0/Asher/code/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"

# local 
# OPENMVG_SFM_BIN = "/home/asher/myVersion/openMVG/build_change/Linux-x86_64-Release"
# CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/asher/myVersion/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"


def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    # H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    F, mask = cv2.findFundamentalMat(ref_points, dst_points, cv2.FM_RANSAC, 3, 0.999)
    mask = mask.flatten()

    print('inlier ratio: ', np.sum(mask)/len(mask))


    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    print("before filtered matches nums is: ", len(keypoints1))
    print("after filtered matches nums is: ", len(matches))
    
    # matches = [cv2.DMatch(i,i,0) for i in range(len(mask))]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)
    
    if len(mask) > 0:
        flag = True

    return img_matches, flag, np.sum(mask)


def load_xfeat_model(params_path):

    params = torch.load(params_path)
    top_k = 4096
    detection_threshold = 0.05
    xfeat = _XFeat(params, top_k=top_k, detection_threshold=detection_threshold)
    
    return xfeat


if __name__ == "__main__":

    starTime = time.time()

    params_path = "/home/asher/myVersion/xfeat/weights/xfeat.pt"

    xfeat = load_xfeat_model(params_path=params_path)

    # import os

    input_dir = "/mnt/c/Users/Asher/Desktop/Data/wheat_alls/29e03514870de0e379b36381"
    output_dir = "/mnt/c/Users/Asher/Desktop/Data/openMVG_SFM/29e03514870de0e379b36381_lowRes_xfeat_2"
    real_img_path = "/mnt/c/Users/Asher/Desktop/Data/wheat_resize/29e03514870de0e379b36381"

    '''
        sift 文件存放的路径 
    '''
    sift_data_path = "/mnt/c/Users/Asher/Desktop/Data/openMVG_SFM/29e03514870de0e379b36381_lowRes_sift"
    image_describer = os.path.join(sift_data_path, "matches", "image_describer.json")
    sfm_data = os.path.join(sift_data_path, "matches", "sfm_data.json")
    assert os.path.isfile(image_describer) and os.path.isfile(sfm_data)

    matches_dir = os.path.join(output_dir, "matches")

    if not os.path.exists(matches_dir):
        os.makedirs(matches_dir)

    import shutil
    shutil.copy(image_describer, matches_dir)
    shutil.copy(sfm_data, matches_dir)

    fileList = os.listdir(real_img_path)
    print("fileList nums is: ", len(fileList))

    image_List = [x for x in fileList if x.endswith(".JPG")]
    image_List = sorted(image_List)
    print("the valid images nums is: ", len(image_List))


    '''
        1. 构建匹配方式
    '''

    # 通过算法的方式得到匹配对
    # from read_exif_data import run_spatial_search_matches_pair
    # matchesPair = run_spatial_search_matches_pair(input_dir, saveFile=matches_dir+"\pairs.txt", topK=10)
    # 通过同一个文件的方式得到匹配对
    sfm_pair = os.path.join(sift_data_path, "matches", "pairs.txt")
    assert os.path.isfile(sfm_pair)
    from pipeline_SFM.sfm_xfeat_steerers_pipeline import read_matches_pairs
    matchesPair = read_matches_pairs(sfm_pair)

    print("the valid image matches length is: ", len(matchesPair))


    '''
        2. 对所有的数据提取特征 
    '''
    idxNameDict = dict()
    featIdxDict = dict()

    for idx, img in enumerate(image_List):
        idxNameDict[idx] = img
        imgFilePath = os.path.join(real_img_path, img)

        assert os.path.isfile(imgFilePath), "the file {} is nots exists.".format(imgFilePath)
        im_Mat = np.array(cv2.imread(imgFilePath))

        feat_output = xfeat.detectAndCompute(im_Mat, top_k = 4096)[0]

        feat_output.update({'image_size': (im_Mat.shape[1], im_Mat.shape[0])})
        
        featIdxDict[idx] = feat_output
        
        prefixName = img.split(".")[0]
        save_feat_file = os.path.join(matches_dir, prefixName+".feat.txt")
        
        featData = feat_output["keypoints"].cpu().numpy()

        ''' debug not save file .... '''
        with open(save_feat_file, 'w') as f:
            for dfeat in featData:
                wStr = str(dfeat[0]) + " " + str(dfeat[1]) + " 1 1\n"
                f.write(wStr)
        print("finished processed images of {} .".format(img))

    print("finished calculate features. ")


    '''    
        2. 对提取的特征进行匹配
    '''

    matches_F_file = os.path.join(matches_dir, "matches.f.txt")
    matches_F_file_w = open(matches_F_file, 'w')

    pairSz = len(matchesPair)

    for idx, pair in enumerate(matchesPair):

        # excute pre and cur image matches.
        # assert len(pair) == 2, "the length of pair must equal to 2."
        idxI, idxJ = pair[0], pair[1]
        output0, output1 = featIdxDict[idxI], featIdxDict[idxJ]

        matchesIdx = xfeat.match_lighterglue_resIdx(output0, output1)

        if len(matchesIdx) < 30:
            continue

        matchesIdxL, matchesIdxR = matchesIdx[:, 0], matchesIdx[:, 1]
            
        mkpts_0 = output0['keypoints'][matchesIdxL].cpu().numpy()
        mkpts_1 = output1['keypoints'][matchesIdxR].cpu().numpy()

        F, mask = cv2.findFundamentalMat(mkpts_0, mkpts_1, cv2.FM_RANSAC, 3, 0.999)
        if mask.sum() < 30:
            continue
        mask = mask.flatten()
        matchesIdxL = matchesIdxL[mask>0]
        matchesIdxR = matchesIdxR[mask>0]
        assert matchesIdxL.shape == matchesIdxR.shape

        writeLineList = []
        
        pairStr = str(idxI) + " " + str(idxJ) + "\n"
        # matches_F_file_w.write(pairStr)
        writeLineList.append(pairStr)
        sz_int = len(matchesIdxL)
        szStr = str(sz_int)+"\n"
        # matches_F_file_w.write(szStr)
        writeLineList.append(szStr)
        for i in range(sz_int):
            feat_idx_L = matchesIdxL[i]
            feat_idx_R = matchesIdxR[i]
            idxStr = str(feat_idx_L) + " " + str(feat_idx_R) + "\n"
            # matches_F_file_w.write(idxStr)
            writeLineList.append(idxStr)
        matches_F_file_w.writelines(writeLineList)

        print("finished process images of {} and {}.".format(idxNameDict[idxI], idxNameDict[idxJ]))
        print("processing index is {}, total nums is: {}".format(idx, pairSz))

    matches_F_file_w.close()
    print("finished all programs.")


    '''
        执行sfm
    '''

    # OPENMVG_SFM_BIN = "/home/asher/myVersion/openMVG/build_change/Linux-x86_64-Release"

    # CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/asher/myVersion/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"


    matches_dir = os.path.join(output_dir, "matches")
    reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")
    camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")


    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    '''
        标定的相机内参：分辨率 2000 x 1500  "-k","862.0;0.0;966.0;0.0;862.0;742.0;0.0;0.0;1.0"
        
    '''

    # Create the reconstruction if not present
    if not os.path.exists(reconstruction_dir):
        os.mkdir(reconstruction_dir)

    print ("6. Do Sequential/Incremental reconstruction")
    pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"), "--sfm_engine", "INCREMENTAL", 
                                "--input_file", matches_dir+"/sfm_data.json", "--match_dir", matches_dir, 
                                "--output_dir", reconstruction_dir] )
    pRecons.wait()

    print ("7. Colorize Structure")
    pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"), 
                                "-i", reconstruction_dir+"/sfm_data.bin", "-o", 
                                os.path.join(reconstruction_dir,"colorized.ply")] )
    pRecons.wait()

    endTime = time.time()

    print("spends times is: min", (endTime-starTime)/60)
