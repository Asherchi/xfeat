import numpy as np
# import imageio as imio
import os
import torch
import tqdm
# import cv2
import matplotlib.pyplot as plt

import cv2
import numpy as np

def filter_matches_by_fundamental_matrix(F, kp1, kp2, matches, threshold):  
    """  
    使用基础矩阵F来筛选匹配点对。  
  
    参数:  
    F -- 基础矩阵（numpy.ndarray，形状为3x3）  
    kp1 -- 第一张图像中的关键点（numpy.ndarray，形状为Nx2或list of cv2.KeyPoint）  
    kp2 -- 第二张图像中的关键点（numpy.ndarray，形状为Nx2或list of cv2.KeyPoint）  
    matches -- 匹配点对列表（list of cv2.DMatch）  
    threshold -- 对极距离阈值  
  
    返回:  
    filtered_matches -- 筛选后的匹配点对列表（list of cv2.DMatch）  
    """  
    # 将关键点转换为numpy数组（如果它们还不是的话）  
    if isinstance(kp1[0], cv2.KeyPoint):  
        kp1 = np.float32([kp.pt for kp in kp1])  
    if isinstance(kp2[0], cv2.KeyPoint):  
        kp2 = np.float32([kp.pt for kp in kp2])  
  
    # 初始化筛选后的匹配列表  
    filtered_matches = []  
  
    # 计算F的转置，因为OpenCV的F是x2^T * F * x1 = 0  
    Ft = F.T  
  
    # 遍历所有匹配  
    for m in matches:  
        # 获取匹配点对的坐标  
        idx1 = m.queryIdx  
        idx2 = m.trainIdx  
        pt1 = kp1[idx1, :]  
        pt2 = kp2[idx2, :]  
  
        # 将pt2转换为齐次坐标  
        pt2_homog = np.hstack((pt2, 1))  
  
        # 计算对极线（在第一张图像上）  
        epipolar_line = np.dot(Ft, pt2_homog)  
        epipolar_line /= epipolar_line[2]  # 归一化  
  
        # 计算对极距离（使用点到直线的距离公式）  
        epipolar_distance = np.abs(pt1[0] * epipolar_line[0] + pt1[1] * epipolar_line[1] + epipolar_line[2])  
  
        # 如果对极距离小于阈值，则保留该匹配  
        if epipolar_distance < threshold:  
            filtered_matches.append(m)  
  
    return filtered_matches  

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    # F, mask = cv2.findFundamentalMat(ref_points, dst_points, cv2.FM_RANSAC, 6, 0.999)
    mask = mask.flatten()

    print('inlier ratio: ', np.sum(mask)/len(mask))

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    # warped_corners = cv2.perspectiveTransform(corners_img1, F)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    # for i in range(len(warped_corners)):
    #     start_point = tuple(warped_corners[i-1][0].astype(int))
    #     end_point = tuple(warped_corners[i][0].astype(int))
    #     cv2.line(img2_with_corners, start_point, end_point, (255, 0, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    # matches = [cv2.DMatch(i,i,0) for i in range(len(mask))]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    print("with F matrix filtered nums is: ", len(matches))

    # matches_filter = filter_matches_by_fundamental_matrix(F=F, kp1=keypoints1, kp2=keypoints2, matches=matches, threshold=0.001)
    
    

    # Draw inlier matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
                                  matchColor=(0, 255, 0), flags=2)

    return img_matches

# !pip install kornia kornia-rs --no-deps # REQUIRED for Lightglue matching

# xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = 4096)

params_path = "/home/asher/data/code/accelerated_features/weights/xfeat.pt"
params = torch.load(params_path)
from modules.xfeat import XFeat as _XFeat
top_k = 4096
detection_threshold = 0.05
xfeat = _XFeat(params, top_k=top_k, detection_threshold=detection_threshold)
# xfeat = ""

#Load some example images
# im1 = np.copy(imio.v2.imread('https://raw.githubusercontent.com/verlab/accelerated_features/main/assets/ref.png')[..., ::-1])
# im2 = np.copy(imio.v2.imread('https://raw.githubusercontent.com/verlab/accelerated_features/main/assets/tgt.png')[..., ::-1])

imgPath = "/mnt/c/Users/Asher/Desktop/Data/feat_match_test/9822e12f10cd92d001282218/"
img1file = os.path.join(imgPath, "XAG218_0045.JPG")
img2file = os.path.join(imgPath, "XAG218_0046.JPG")
assert os.path.isfile(img1file), "the file {} is nots exists.".format(img1file)
assert os.path.isfile(img2file), "the file {} is nots exists.".format(img2file)

# im1_np = np.array(cv2.imread("/home/asher/data/test_data/XAG69a_0159.JPG"))
# im2_np = np.array(cv2.imread("/home/asher/data/test_data/XAG69a_0160.JPG"))
im1_np = np.array(cv2.imread(img1file))
im2_np = np.array(cv2.imread(img2file))

width = 800
height = 600
im1_np = cv2.resize(im1_np, (width, height))
im2_np = cv2.resize(im2_np, (width, height))

outputs = xfeat.match_xfeat_star(im1_np, im2_np, min_cossim=0.9, top_k=4096)

# outputs2= xfeat.match_xfeat(im1_np, im2_np)

feat1 = outputs[0]
feat2 = outputs[1]

print("origin matches nums is: ", len(feat1))

# keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in feat1]
# keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in feat2]

# matches = [cv2.DMatch(i,i,0) for i in range(len(feat1))]

# matchersImg = cv2.drawMatches(im1_np, keypoints1, im2_np, keypoints2, matches, None,
#                               matchColor=(0, 255, 0), flags=2)

# plt.imsave("Figure_asher.png", matchersImg[..., ::-1])
# cv2.imwrite("matchers_800x600.png", matchersImg)

canvas = warp_corners_and_draw_matches(feat1, feat2, im1_np, im2_np)
# plt.figure(figsize=(12,12))
# plt.imshow(canvas[..., ::-1]), plt.show()
cv2.imwrite("xfeat_dense.png", canvas)