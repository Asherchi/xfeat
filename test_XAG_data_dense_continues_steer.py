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


@torch.inference_mode()
def match_xfeat_with_steerer(img1, img2, top_k = None, min_cossim = -1):
    """
        Simple extractor and MNN matcher.
        For simplicity it does not support batched mode due to possibly different number of kpts.
        input:
            img1 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
            img2 -> torch.Tensor (1,C,H,W) or np.ndarray (H,W,C): grayscale or rgb image.
            top_k -> int: keep best k features
        returns:
            mkpts_0, mkpts_1 -> np.ndarray (N,2) xy coordinate matches from image1 to image2
    """
    if top_k is None: top_k = xfeat.top_k
    img1 = xfeat.parse_input(img1)
    img2 = xfeat.parse_input(img2)

    out1 = xfeat.detectAndCompute(img1, top_k=top_k)[0]
    out2 = xfeat.detectAndCompute(img2, top_k=top_k)[0]

    idxs0, idxs1 = xfeat.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)
    rot1to2 = 0
    for r in range(1, 4):
        out1['descriptors'] = torch.nn.functional.normalize(steerer(out1['descriptors']), dim=-1)
        new_idxs0, new_idxs1 = xfeat.match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim)
        if len(new_idxs0) > len(idxs0):
            idxs0 = new_idxs0
            idxs1 = new_idxs1
            rot1to2 = r

    return out1['keypoints'][idxs0].cpu().numpy(), out2['keypoints'][idxs1].cpu().numpy(), rot1to2



STEER_PERMUTATIONS = [
    torch.arange(64).reshape(4, 16).roll(k, dims=0).reshape(64)
    for k in range(4)
]

@torch.inference_mode()
def match_xfeat_star_with_permutation_steerer(im_set1, im_set2, top_k = None, min_cossim_coarse=-1):
    """
        Extracts coarse feats, then match pairs and finally refine matches, currently supports batched mode.
        input:
            im_set1 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
            im_set2 -> torch.Tensor(B, C, H, W) or np.ndarray (H,W,C): grayscale or rgb images.
            top_k -> int: keep best k features
        returns:
            matches -> List[torch.Tensor(N, 4)]: List of size B containing tensor of pairwise matches (x1,y1,x2,y2)
    """
    if top_k is None: top_k = xfeat.top_k

    B = im_set1.shape[0] if len(im_set1.shape) == 4 else 1
    if B > 1:
        raise NotImplementedError("TODO: Batched dense matching with steerer")

    im_set1 = xfeat.parse_input(im_set1)
    im_set2 = xfeat.parse_input(im_set2)

    #Compute coarse feats
    out1 = xfeat.detectAndComputeDense(im_set1, top_k=top_k)
    out2 = xfeat.detectAndComputeDense(im_set2, top_k=top_k)

    rot1to2 = 0
    idxs_list = xfeat.batch_match(out1['descriptors'], out2['descriptors'], min_cossim=min_cossim_coarse)
    for r in range(1, 4):
        new_idxs_list = xfeat.batch_match(out1['descriptors'][..., STEER_PERMUTATIONS[r]], out2['descriptors'], min_cossim=min_cossim_coarse)
        if len(new_idxs_list[0][0]) > len(idxs_list[0][0]):
            idxs_list = new_idxs_list
            rot1to2 = r

    out2['descriptors'] = out2['descriptors'][..., STEER_PERMUTATIONS[-rot1to2]]  # align to first image for refinement MLP

    #Refine coarse matches
    #this part is harder to batch, currently iterate
    matches = xfeat.refine_matches(out1, out2, matches=idxs_list, batch_idx=0)

    return matches[:, :2].cpu().numpy(), matches[:, 2:].cpu().numpy(), rot1to2

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

steer_xfeat_pth = "/home/asher/data/code/accelerated_features/weights/xfeat_learn_steer.pth"

# params_path = "/home/asher/data/code/accelerated_features/weights/xfeat.pt"
params = torch.load(steer_xfeat_pth, map_location='cpu')
from modules.xfeat import XFeat as _XFeat
top_k = 4096
detection_threshold = 0.05
xfeat = _XFeat(params, top_k=top_k, detection_threshold=detection_threshold)

steer_pth = "/home/asher/data/code/accelerated_features/weights/xfeat_learn_steer_steerer.pth"
steerer = torch.nn.Linear(64, 64, bias=False)
steerer.weight.data = torch.load(steer_pth, map_location='cpu')['weight'][..., 0, 0]
steerer.eval()

imgPath = "/mnt/c/Users/Asher/Desktop/Data/feat_match_test/7ae2960a4176360b8690d8df/"
img1file = os.path.join(imgPath, "XAG8df_0132.JPG")
img2file = os.path.join(imgPath, "XAG8df_0133.JPG")
img3file = os.path.join(imgPath, "XAG8df_0134.JPG")

assert os.path.isfile(img1file), "the file {} is nots exists.".format(img1file)
assert os.path.isfile(img2file), "the file {} is nots exists.".format(img2file)
assert os.path.isfile(img3file), "the file {} is nots exists.".format(img3file)

outputPath = os.path.join(imgPath, "steer_xfeat_continues")
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


''' 下面是传统的xfeat的dense方法  '''
# start_time = time.perf_counter()

# outputs = xfeat.match_xfeat_star(im1, im2, min_cossim=0.9, top_k=4096)

# mkpts_0 = outputs[0]
# mkpts_1 = outputs[1]

# outputs = xfeat.match_xfeat_star(im2, im3, min_cossim=0.9, top_k=4096)

# mkpts_11 = outputs[0]
# mkpts_2 = outputs[1]

# end_time = time.perf_counter()


''' 下面是xfeat的方法 + steer的方法 '''
# mkpts_0, mkpts_1, rot1to2 = match_xfeat_star_with_permutation_steerer(im1, im2, top_k = 8000, min_cossim_coarse=.9)

# mkpts_11, mkpts_2, rot1to2 = match_xfeat_star_with_permutation_steerer(im1, im2, top_k = 8000, min_cossim_coarse=.9)

# print("the algorithm excute time is: ", (end_time-start_time) * 1000 )

''' 下面是sparse的xfeat的方法 + steer '''
mkpts_0, mkpts_1, rot1to12 = match_xfeat_with_steerer(im1, im2, top_k = 4096, min_cossim=.9)
mkpts_11, mkpts_2, rot1to23 = match_xfeat_with_steerer(im2, im3, top_k = 4096, min_cossim=.9)


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
print("inliner ratios is: ", continues_nums_F / (continues_nums + 1e-8))
print("valid nums ratios is: ", continues_nums_F * 2 / (validSum + 1e-8))


# matchedImg12 = draw_two_images_matches(mkpts_0, mkpts_1, im1, im2, matched_12)
# matchedImg23 = draw_two_images_matches(mkpts_11, mkpts_2, im2, im3, matched_23)

# matchedImg12Name = os.path.join(outputPath, "matchedImg12.png")
# matchedImg23Name = os.path.join(outputPath, "matchedImg23.png")
# cv2.imwrite(matchedImg12Name, matchedImg12)
# cv2.imwrite(matchedImg23Name, matchedImg23)

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

print("输出路径为: ", finalMatchImgName_F)


