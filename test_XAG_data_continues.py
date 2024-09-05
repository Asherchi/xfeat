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

    return img_matches



def draw_two_images_matches(ref_points, dst_points, img1, img2, matches):
    # Calculate the Homography matrix

    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches


def draw_three_image_matches(points1, points2, points3, img1, img2, img3, matches12, matches23):

    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    h3, w3, _ = img3.shape

    img_hconcat = cv2.hconcat((img1, img2, img3))

    for match in matches12:

        idx1, idx2 = match.queryIdx, match.trainIdx
        pos1, pos2 = points1[idx1], points2[idx2]
        x1, y1 = int(pos1[0]), int(pos1[1])
        x2, y2 = int(pos2[0] + w1), int(pos2[1])
        cv2.circle(img_hconcat, (x1, y1), 2, (0, 0, 255), 2)
        cv2.circle(img_hconcat, (x2, y2), 2, (0, 0, 255), 2)
        cv2.line(img_hconcat, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    
    for match in matches23:

        idx1, idx2 = match.queryIdx, match.trainIdx
        pos1, pos2 = points2[idx1], points3[idx2]
        x1, y1 = int(pos1[0] + w1 ), int(pos1[1])
        x2, y2 = int(pos2[0] + w1 + w2), int(pos2[1])
        # cv2.circle(img_hconcat, (x1, y1), 2, (0, 0, 255), 2) // 因为前一个步骤已经在图上画了一个圈 
        cv2.circle(img_hconcat, (x2, y2), 2, (0, 0, 255), 2)
        cv2.line(img_hconcat, (x1, y1), (x2, y2), (255, 0, 0), 2)


    return img_hconcat

    



params_path = "/home/asher/data/code/accelerated_features/weights/xfeat.pt"
params = torch.load(params_path)
from modules.xfeat import XFeat as _XFeat
top_k = 4096
detection_threshold = 0.05
xfeat = _XFeat(params, top_k=top_k, detection_threshold=detection_threshold)
# xfeat = ""

imgPath = "/mnt/c/Users/Asher/Desktop/Data/good_texture/96396da202d23e02f9a7e119/"
img1file = os.path.join(imgPath, "XAG119_0019.JPG")
img2file = os.path.join(imgPath, "XAG119_0020.JPG")
img3file = os.path.join(imgPath, "XAG119_0021.JPG")

assert os.path.isfile(img1file), "the file {} is nots exists.".format(img1file)
assert os.path.isfile(img2file), "the file {} is nots exists.".format(img2file)
assert os.path.isfile(img3file), "the file {} is nots exists.".format(img3file)

outputPath = os.path.join(imgPath, "xfeat_continues")
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

im1 = np.array(cv2.imread(img1file))
im2 = np.array(cv2.imread(img2file))
im3 = np.array(cv2.imread(img3file))

width = 800
height = 600
im1 = cv2.resize(im1, (width, height))
im2 = cv2.resize(im2, (width, height))
im3 = cv2.resize(im3, (width, height))

img_hconcat = cv2.hconcat((im1, im2, im3))
concat_img_file = os.path.join(outputPath, "concat.png")
cv2.imwrite(concat_img_file, img_hconcat)

start_time = time.perf_counter()

output0 = xfeat.detectAndCompute(im1, top_k = 4096)[0]
output1 = xfeat.detectAndCompute(im2, top_k = 4096)[0]
output2 = xfeat.detectAndCompute(im3, top_k = 4096)[0]

#Update with image resolution (required)
output0.update({'image_size': (im1.shape[1], im1.shape[0])})
output1.update({'image_size': (im2.shape[1], im2.shape[0])})
output2.update({'image_size': (im3.shape[1], im3.shape[0])})

mkpts_0, mkpts_1 = xfeat.match_lighterglue(output0, output1)
mkpts_11, mkpts_2 = xfeat.match_lighterglue(output0, output1)

end_time = time.perf_counter()

print("the algorithm excute time is: ", (end_time-start_time) * 1000 )

print("no filter image match of 1 and 2 nums is: ", len(mkpts_0))
print("no filter image match of 2 and 3 nums is: ", len(mkpts_11))

'''
    判断mkpts_1和mkpts_11的关系

'''
valIdx12Dict = dict()
for idx, arr in enumerate(mkpts_1):
    valIdx12Dict[tuple(arr)] = idx
valIdx23Dict = dict()
for idx, arr in enumerate(mkpts_11):
    valIdx23Dict[tuple(arr)] = idx

F12, mask12 = cv2.findFundamentalMat(mkpts_0, mkpts_1, cv2.FM_RANSAC, 3, 0.999, maxIters=1000)
mask12 = mask12.flatten()

F23, mask23 = cv2.findFundamentalMat(mkpts_11, mkpts_2, cv2.FM_RANSAC, 3, 0.999, maxIters=1000)
mask23 = mask23.flatten()

validSum = np.sum(mask12) + np.sum(mask23)

print("F12 is: ", F12)
print("F23 is: ", F23)

matched_12, matched_23 = [], []
matched_12_F, matched_23_F = [], []

continues_nums = 0
continues_nums_F = 0
mkpt_list = [list(arr) for arr in mkpts_1]
for sub_arr in mkpts_11:
    sub_list = list(sub_arr)
    if sub_list in mkpt_list:
        _12idx = valIdx12Dict[tuple(sub_arr)]
        _23idx = valIdx23Dict[tuple(sub_arr)]
        continues_nums += 1
        matched_12.append(cv2.DMatch(_12idx, _12idx, 0))
        matched_23.append(cv2.DMatch(_12idx, _23idx, 0))

        if mask12[_12idx] and mask23[_23idx]:
            matched_12_F.append(cv2.DMatch(_12idx, _12idx, 0))
            matched_23_F.append(cv2.DMatch(_12idx, _23idx, 0))
            continues_nums_F+=1

print("continues nums is: ", continues_nums)
print("after F filter nums is: ", continues_nums_F)
print("inliner ratios is: ", continues_nums_F / (continues_nums + 1e-8) * 100 )
print("valid nums ratios is: ", continues_nums_F * 2 / (validSum + 1e-8) * 100 )

matchedImg12 = draw_two_images_matches(mkpts_0, mkpts_1, im1, im2, matched_12)
matchedImg23 = draw_two_images_matches(mkpts_11, mkpts_2, im2, im3, matched_23)

matchedImg12Name = os.path.join(outputPath, "matchedImg12.png")
matchedImg23Name = os.path.join(outputPath, "matchedImg23.png")
cv2.imwrite(matchedImg12Name, matchedImg12)
cv2.imwrite(matchedImg23Name, matchedImg23)

finalMatchImg = draw_three_image_matches(points1=mkpts_0, points2=mkpts_1, points3=mkpts_2,
                                         img1=im1, img2=im2, img3=im3, matches12=matched_12,
                                         matches23=matched_23)

finalMatchImgName = os.path.join(outputPath, "finalMatchImg.png")
cv2.imwrite(finalMatchImgName, finalMatchImg)


finalMatchImg_F = draw_three_image_matches(points1=mkpts_0, points2=mkpts_1, points3=mkpts_2,
                                         img1=im1, img2=im2, img3=im3, matches12=matched_12_F,
                                         matches23=matched_23_F)

finalMatchImgName_F = os.path.join(outputPath, "finalMatchImg_F.png")
cv2.imwrite(finalMatchImgName_F, finalMatchImg_F)


