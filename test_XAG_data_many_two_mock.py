import numpy as np
# import imageio as imio
import os
import torch
import tqdm
import cv2
import matplotlib.pyplot as plt

import cv2
import numpy as np
import time

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



params_path = "/home/asher/data/code/accelerated_features/weights/xfeat.pt"
params = torch.load(params_path)
from modules.xfeat import XFeat as _XFeat
top_k = 4096
detection_threshold = 0.05
xfeat = _XFeat(params, top_k=top_k, detection_threshold=detection_threshold)


import os

image_path = "/mnt/c/Users/Asher/Desktop/Data/PV_wheat_field/29e03514870de0e379b36381"
output_dir = "/mnt/c/Users/Asher/Desktop/Data/openMVG_SFM/29e03514870de0e379b36381_xfeat/matches"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

fileList = os.listdir(image_path)
print("fileList nums is: ", len(fileList))

image_List = [x for x in fileList if x.endswith(".JPG")]
image_List = sorted(image_List)
print("the valid images nums is: ", len(image_List))

passTwoMatches = 0
passFfilterMatches = 0
all_matches_nums = 0
all_filter_matches_num = 0
min_match_nums = 4096
min_match_F_nums = 4096
max_nums = 0

'''
    1. 构建匹配方式
'''

# 暴力匹配
num_len = len(image_List)
matchesPair = []
for i in range(num_len):

    for j in range(i+1, num_len):
        pair = [i, j]
        matchesPair.append(pair)

print("exhaustive matches nums is: ", len(matchesPair))

# 顺序匹配


# 位置匹配

'''
    2. 对所有的数据提取特征 
'''
idxNameDict = dict()
featIdxDict = dict()

for idx, img in enumerate(image_List):
    idxNameDict[idx] = img
    imgFilePath = os.path.join(image_path, img)

    assert os.path.isfile(imgFilePath), "the file {} is nots exists.".format(imgFilePath)
    im_Mat = np.array(cv2.imread(imgFilePath))

    feat_output = xfeat.detectAndCompute(im_Mat, top_k = 4096)[0]

    feat_output.update({'image_size': (im_Mat.shape[1], im_Mat.shape[0])})
    
    featIdxDict[idx] = feat_output
    
    prefixName = img.split(".")[0]
    save_feat_file = os.path.join(output_dir, prefixName+".feat.txt")
    
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

matches_F_file = os.path.join(output_dir, "matches.f.txt")
matches_F_file_w = open(matches_F_file, 'w')

pairSz = len(matchesPair)

for idx, pair in enumerate(matchesPair):

    # excute pre and cur image matches.
    # assert len(pair) == 2, "the length of pair must equal to 2."
    idxI, idxJ = pair[0], pair[1]
    output0, output1 = featIdxDict[idxI], featIdxDict[idxJ]

    matchesIdx = xfeat.match_lighterglue_resIdx(output0, output1)

    if len(matchesIdx) < 50:
        continue

    matchesIdxL, matchesIdxR = matchesIdx[:, 0], matchesIdx[:, 1]
		
    mkpts_0 = output0['keypoints'][matchesIdxL].cpu().numpy()
    mkpts_1 = output1['keypoints'][matchesIdxR].cpu().numpy()

    F, mask = cv2.findFundamentalMat(mkpts_0, mkpts_1, cv2.FM_RANSAC, 3, 0.999)
    if mask.sum() < 50:
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


''' as following are used for valid thr methods effective. '''

    # mkpts_00 = output0['keypoints'][matchesIdxL].cpu().numpy()
    # mkpts_11 = output1['keypoints'][matchesIdxR].cpu().numpy()

    # im1Name = idxNameDict[idxI]
    # im2Name = idxNameDict[idxJ]
    # file1 = os.path.join(image_path, im1Name)
    # file2 = os.path.join(image_path, im2Name)
    # im1 = cv2.imread(file1)
    # im2 = cv2.imread(file2)
    # canvas, F_flag, validNums = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)

    # cv2.imwrite("valid_mathods_effective.png", canvas)

    # print("finished the process {} and {}.".format(idxNameDict[idxI], idxNameDict[idxJ]))


