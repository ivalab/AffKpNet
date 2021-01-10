# this script write kp into file
import sys
import math
import cv2
import numpy as np
import os
import cv2.aruco as aruco
import scipy.io as sio

from arucoTag import get_multi_M_CL
from kp_utils import read_kp_profile



########################
# file path
########################
# object folder
OBJECT = 'spoon_05'
# pic root
PIC_ROOT = '../../images/complementary/official/images'
# ground truth save path
SAVE_ROOT = '../../images/complementary/official/annotations_kp'
# keypoint profile path
PROFILE_ROOT = '../kp_profiles'


########################
# change mode
########################
DEBUG = True
VISUALIZE = False


########################
# read profile (keypoints)
########################
kp_profile_path = os.path.join(PROFILE_ROOT, OBJECT + '_profile.txt' )
aff, offset, keypoint_list, aff2, offset2 ,keypoint_list2 = read_kp_profile(kp_profile_path)


for idx, keypoint in enumerate(keypoint_list):
    keypoint_list[idx] = [a + b for a, b in zip(keypoint, offset)]

for idx, keypoint in enumerate(keypoint_list2):
    keypoint_list2[idx] = [a + b for a, b in zip(keypoint, offset2)]



###### DO NOT MODIFY BELOW #######
# Realsense D435 depth camera intrinsic matrix
#cameraMatrix = np.array([[613.8052368164062, 0.0, 328.09918212890625],
#                         [0.0, 613.8528442382812, 242.4539337158203],
#                         [0.0, 0.0, 1.0]])
#distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Caltech calibration
cameraMatrix = np.array([[607.47165, 0.0, 325.90064],
                         [0.0, 606.30420, 240.91934],
                         [0.0, 0.0, 1.0]])
distCoeffs = np.array([0.08847,   -0.04283,   0.00134,   -0.00102,  0.00000])

depth_scale = 0.001

# default transformation matrix from camera to aruco tag
default_M_CL = np.array([[8.05203923e-04,  -9.98258274e-01,  5.89895796e-02, 2.11182116e-02],
                         [-7.25197650e-01, -4.11996435e-02, -6.87307033e-01, 1.19383476e-01],
                         [6.88540282e-01,  -4.22256822e-02, -7.23967728e-01, 5.68361874e-01],
                         [0.00000000e+00,   0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])


PIC_PATH = os.path.join(PIC_ROOT, OBJECT)
if not os.path.exists(os.path.join(SAVE_ROOT, OBJECT)):
    os.mkdir(os.path.join(SAVE_ROOT, OBJECT))
SAVE_PATH = os.path.join(SAVE_ROOT, OBJECT)

# for each degree
for count in range(180, 360):
    #if count < 18:
    #   continue
    print("generating groundtruth for no.{0:3d}".format(count))
    # for file in file_list:
    img = cv2.imread(os.path.join(PIC_PATH, 'rgb_' + str(count) + '.png'))
    depth_img = np.load(os.path.join(PIC_PATH, 'depth_' + str(count) + '.npy'))

    kp_file_path =os.path.join(SAVE_PATH, str(count) + '_keypoint.txt')
    kp_file = open(kp_file_path, "w")
    kp_file.write(aff + '\n')

    # Show images
    if DEBUG:
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', img)
        cv2.waitKey(0)

    # detect four aruco tags
    gray = img.astype(np.uint8)
    mask_kp = img.copy()
    for keypoint in keypoint_list:
        results = get_multi_M_CL(gray, img, keypoint, VISUALIZE)  # size should be four

        avg_diagonal_pixel = []
        for result in results:
            diagonal_pixel = result['diagonal_pixel']
            avg_diagonal_pixel.append(diagonal_pixel)
            #cv2.circle(mask_kp, (int(diagonal_pixel[0]), int(diagonal_pixel[1])), radius=0, color=(255, 0, 0), thickness=2)

        avg_diagonal_pixel = sum(avg_diagonal_pixel)/len(avg_diagonal_pixel)
        # write to file
        kp_file.write(str(round(avg_diagonal_pixel[0], 4)) + ' ' + str(round(avg_diagonal_pixel[1], 4)) + '\n')
        cv2.circle(mask_kp, (int(avg_diagonal_pixel[0]), int(avg_diagonal_pixel[1])), radius=0, color=(0, 255, 0),
                   thickness=2)

    if keypoint_list2:
        kp_file.write(aff2 + '\n')
        for keypoint in keypoint_list2:
            results = get_multi_M_CL(gray, img, keypoint, VISUALIZE)  # size should be four

            avg_diagonal_pixel = []
            for result in results:
                diagonal_pixel = result['diagonal_pixel']
                avg_diagonal_pixel.append(diagonal_pixel)
                #cv2.circle(mask_kp, (int(diagonal_pixel[0]), int(diagonal_pixel[1])), radius=0, color=(255, 0, 0), thickness=2)

            avg_diagonal_pixel = sum(avg_diagonal_pixel)/len(avg_diagonal_pixel)
            kp_file.write(str(round(avg_diagonal_pixel[0],4)) + ' ' + str(round(avg_diagonal_pixel[1],4)) + '\n')
            cv2.circle(mask_kp, (int(avg_diagonal_pixel[0]), int(avg_diagonal_pixel[1])), radius=0, color=(0, 0, 255),
                       thickness=2)

    if DEBUG:
        cv2.namedWindow('center point', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('center point', mask_kp)
        cv2.waitKey(0)

    kp_file.close()


