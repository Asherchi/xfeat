

# 172.31.178.53
OPENMVG_SFM_BIN = "/home/xavision/nnd_storage_0/Asher/code/openMVG/build/Linux-x86_64-Release"
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/xavision/nnd_storage_0/Asher/code/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"

# local 
# OPENMVG_SFM_BIN = "/home/asher/myVersion/openMVG/build_change/Linux-x86_64-Release"
# CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/asher/myVersion/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"



import os
import subprocess
import sys
import time
import cv2

def resize_image(imgPath, savePath):
   
    # imgPath = "/mnt/c/Users/Asher/Desktop/Data/wheat_alls/5f27de0faace21a85b35739d"
    # savePath = "/mnt/c/Users/Asher/Desktop/Data/wheat_resize/5f27de0faace21a85b35739d"
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    imgList = os.listdir(imgPath)


    for fileName in imgList:
        if fileName.split(".")[-1] != "JPG":
            continue
        imgFile = os.path.join(imgPath, fileName)
        imgMat = cv2.imread(imgFile)

        imgMat = cv2.resize(imgMat, (800, 600))
        saveFile = os.path.join(savePath, fileName)
        cv2.imwrite(saveFile, imgMat)

        print("finished img is: ", fileName)

    print("finished.")


starTime = time.time()

'''
    input_dir: 原始分辨率的数据的路径
    output_dir: 输出的路径
    real_img_path: 真实图像的路径 
'''

input_dir = "/mnt/c/Users/Asher/Desktop/Data/wheat_alls/5f27de0faace21a85b35739d"
output_dir = "/mnt/c/Users/Asher/Desktop/Data/openMVG_SFM/5f27de0faace21a85b35739d_lowRes_sift"
real_img_path = "/mnt/c/Users/Asher/Desktop/Data/wheat_resize/5f27de0faace21a85b35739d"

matches_dir = os.path.join(output_dir, "matches")
reconstruction_dir = os.path.join(output_dir, "reconstruction_sequential")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

if not os.path.exists(real_img_path):
   os.makedirs(real_img_path)
   resize_image(input_dir, real_img_path)

print ("      output_dir : ", output_dir)

# Create the ouput/matches folder if not present
if not os.path.exists(output_dir):
  os.mkdir(output_dir)
if not os.path.exists(matches_dir):
  os.mkdir(matches_dir)

print ("1. Intrinsics analysis")
'''
    2000 x 1500: "862.0;0.0;966.0;0.0;862.0;742.0;0.0;0.0;1.0"
    800  x 600 : "344.8;0.0;386.4;0.0;344.8;296.8;0.0;0.0;1.0"    
'''
pIntrisics = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"),  
                                "-i", real_img_path, "-o", matches_dir, "-d", camera_file_params, 
                                "-k","344.8;0.0;386.4;0.0;344.8;296.8;0.0;0.0;1.0" ] )
pIntrisics.wait()

print ("2. Compute features")
pFeatures = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"),  
                               "-i", matches_dir+"/sfm_data.json", "-o", matches_dir, "-m", "SIFT"] )
pFeatures.wait()

print ("3. Compute matching pairs")

from read_exif_data import run_spatial_search_matches_pair
run_spatial_search_matches_pair(imagesPath=input_dir, saveFile=matches_dir+"/pairs.txt", topK=10)

print ("4. Compute matches")
pMatches = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"),  
                              "-i", matches_dir+"/sfm_data.json", "-p", matches_dir+ "/pairs.txt", 
                              "-o", matches_dir + "/matches.putative.txt" ] )
pMatches.wait()

print ("5. Filter matches" )
pFiltering = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"),
                                 "-i", matches_dir+"/sfm_data.json", "-m", matches_dir+"/matches.putative.txt" ,
                                   "-g" , "f" , "-o" , matches_dir+"/matches.f.txt" ] )
pFiltering.wait()

# Create the reconstruction if not present
if not os.path.exists(reconstruction_dir):
    os.mkdir(reconstruction_dir)

print ("6. Do Sequential/Incremental reconstruction")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"), "--sfm_engine", 
                             "INCREMENTAL", "--input_file", matches_dir+"/sfm_data.json", "--match_dir",
                               matches_dir, "--output_dir", reconstruction_dir] )
pRecons.wait()

print ("7. Colorize Structure")
pRecons = subprocess.Popen( [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"),  
                             "-i", reconstruction_dir+"/sfm_data.bin", "-o", 
                             os.path.join(reconstruction_dir,"colorized.ply")] )
pRecons.wait()

endTime = time.time()

print("spends time is: ", (endTime-starTime)/60)

