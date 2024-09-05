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


params_path = "/home/asher/data/code/accelerated_features/weights/xfeat.pt"
params = torch.load(params_path)
from modules.xfeat import XFeat as _XFeat
top_k = 4096
detection_threshold = 0.05
xfeat = _XFeat(params, top_k=top_k, detection_threshold=detection_threshold)

imgPath = "/mnt/c/Users/Asher/Desktop/Data/feat_match_test/9822e12f10cd92d001282218/"
img1file = os.path.join(imgPath, "XAG218_0045.JPG")
img2file = os.path.join(imgPath, "XAG218_0046.JPG")
assert os.path.isfile(img1file), "the file {} is nots exists.".format(img1file)
assert os.path.isfile(img2file), "the file {} is nots exists.".format(img2file)

outputPath = os.path.join(imgPath, "xfeat")
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

im1 = np.array(cv2.imread(img1file))
im2 = np.array(cv2.imread(img2file))

width = 800
height = 600
im1 = cv2.resize(im1, (width, height))
im2 = cv2.resize(im2, (width, height))

img_hconcat = cv2.hconcat((im1, im2))
concat_img_file = os.path.join(outputPath, "concat.png")


mkpts_0, mkpts_1, rot1to2 = match_xfeat_star_with_permutation_steerer(im1, im2, top_k = 8000, min_cossim_coarse=.9)

print(f"Number 90 deg rotations from first image to second: {rot1to2}")

canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
# plt.figure(figsize=(12,12))
# plt.imshow(canvas[..., ::-1]), plt.show()

cv2.imwrite("steer.png", canvas)

# cv2.imwrite(concat_img_file, img_hconcat)

# start_time = time.perf_counter()

# output0 = xfeat.detectAndCompute(im1, top_k = 4096)[0]
# output1 = xfeat.detectAndCompute(im2, top_k = 4096)[0]

# #Update with image resolution (required)
# output0.update({'image_size': (im1.shape[1], im1.shape[0])})
# output1.update({'image_size': (im2.shape[1], im2.shape[0])})

# mkpts_0, mkpts_1 = xfeat.match_lighterglue(output0, output1)

# end_time = time.perf_counter()

# print("the algorithm excute time is: ", (end_time-start_time) * 1000 )

# matches_ = [cv2.DMatch(i,i,0) for i in range(len(mkpts_0))]

# keypoints1_ = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts_0]
# keypoints2_ = [cv2.KeyPoint(p[0], p[1], 5) for p in mkpts_1]

# matchersImg = cv2.drawMatches(im1, keypoints1_, im2, keypoints2_, matches_, None,
#                               matchColor=(0, 255, 0), flags=2)

# straight_match_file = os.path.join(outputPath, "straight_match.png")
# cv2.imwrite(straight_match_file, matchersImg)


# canvas = warp_corners_and_draw_matches(mkpts_0, mkpts_1, im1, im2)
# # plt.figure(figsize=(12,12))
# # plt.imshow(canvas[..., ::-1]), plt.show()
# # plt.imsave("Figure_asher.png", matchersImg[..., ::-1])
# save_match_file = os.path.join(outputPath, "xfeat_sparse_match_F_600x800.png")
# cv2.imwrite(save_match_file, canvas)

# print("save file in: ", save_match_file)

pass