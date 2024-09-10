
import os
import cv2
import numpy as np


def rotate_images_180(images_list, imgPath, savePath):

    for img in images_list:
        file = os.path.join(imgPath, img)
        assert os.path.isfile(file)
        imgMat = cv2.imread(file)
        imgMat_r = np.rot90(imgMat, 2)
        saveFile = os.path.join(savePath, img)
        cv2.imwrite(saveFile, imgMat_r)

    return 


if __name__ == "__main__":

    imgPath = "/mnt/c/Users/Asher/Desktop/Data/relative_frames/7ae2960a4176360b8690d8df_2"
    savePath = "/mnt/c/Users/Asher/Desktop/Data/relative_frames/7ae2960a4176360b8690d8df_2/rotate180"
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    imgList = ["XAG8df_0149.JPG"]
    rotate_images_180(images_list=imgList, imgPath=imgPath, savePath=savePath)
    print("finished.")