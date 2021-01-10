import sys
import math
import cv2
import numpy as np
import os
import cv2.aruco as aruco
import scipy.io as sio


# Realsense D435 depth camera intrinsic matrix
cameraMatrix = np.array([[613.8052368164062, 0.0, 328.09918212890625],
                         [0.0, 613.8528442382812, 242.4539337158203],
                         [0.0, 0.0, 1.0]])
distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

depth_scale = 0.001

# default transformation matrix from camera to aruco tag
default_M_CL = np.array([[8.05203923e-04,  -9.98258274e-01,  5.89895796e-02, 2.11182116e-02],
                         [-7.25197650e-01, -4.11996435e-02, -6.87307033e-01, 1.19383476e-01],
                         [6.88540282e-01,  -4.22256822e-02, -7.23967728e-01, 5.68361874e-01],
                         [0.00000000e+00,   0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

# transformation matrix from 3rd aruco tag to 4th aruco tag
M_34 = np.array([[ 9.99798700e-01,  1.45612018e-02,  1.38033000e-02,  4.84811426e-01],
                 [-1.42374489e-02,  9.99627837e-01, -2.32697797e-02,  2.51517701e-03],
                 [-1.41369989e-02,  2.30685717e-02,  9.99633926e-01, -4.74676209e-04],
                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

M_43 = np.array([[ 0.99825446,  -0.04546784,  -0.03769243,  -0.48706749],
                 [ 0.04625246,   0.99872528,   0.020212,    -0.01028885],
                 [ 0.03672539,  -0.02192008,  0.99908496, -0.00448],
                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

# transformation matrix from 3rd aruco tag to 5th aruco tag
M_35 = np.array([[ 0.99996112,  0.00489006, -0.00733781,  0.48755225],
                 [-0.00496643,  0.99993333, -0.01042471, -0.28464153],
                 [ 0.00728634,  0.01046074,  0.99991874,  0.01161537],
                 [ 0.,          0.,          0.,          1.        ]])

# transformation matrix from 3rd aruco tag to 6th aruco tag
M_36 = np.array([[ 9.98172337e-01, -4.04213963e-02, -4.49232255e-02,  4.86258006e-04],
                 [ 4.09087649e-02,  9.99113016e-01,  9.98268124e-03, -2.92464006e-01],
                 [ 4.44798654e-02, -1.18021899e-02,  9.98940564e-01,  3.47916976e-03],
                 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])


M_dict = {'34': np.linalg.inv(M_43),
          '35': M_35,
          '36': M_36,
          '43': M_43,
          '45': np.linalg.inv(M_34).dot(M_35),
          '46': np.linalg.inv(M_34).dot(M_36),
          '53': np.linalg.inv(M_35),
          '54': np.linalg.inv(M_35).dot(M_34),
          '56': np.linalg.inv(M_35).dot(M_36),
          '63': np.linalg.inv(M_36),
          '64': np.linalg.inv(M_36).dot(M_34),
          '65': np.linalg.inv(M_36).dot(M_35)}

# transformation matrix from center o to aruco tag to 3rd aruco tag
M_a3O = np.array([[ 1.0,  0.0,  0.0,  0.242],
                 [  0.0,  1.0,  0.0,  -0.142],
                 [  0.0,  0.0,  1.0,  0.0],
                 [  0.0,  0.0,  0.0,  1.0]])
M_a4O = np.array([[ 1.0,  0.0,  0.0,  -0.242],
                 [  0.0,  1.0,  0.0,  -0.142],
                 [  0.0,  0.0,  1.0,  0.0],
                 [  0.0,  0.0,  0.0,  1.0]])
M_a5O = np.array([[ 1.0,  0.0,  0.0,  -0.242],
                 [  0.0,  1.0,  0.0,  0.142],
                 [  0.0,  0.0,  1.0,  0.0],
                 [  0.0,  0.0,  0.0,  1.0]])
M_a6O = np.array([[ 1.0,  0.0,  0.0,  0.242],
                 [  0.0,  1.0,  0.0,  0.142],
                 [  0.0,  0.0,  1.0,  0.0],
                 [  0.0,  0.0,  0.0,  1.0]])


#####################################################
# Function: get the T matrix from camera to aruco tag
#####################################################
def get_multi_M_CL(gray, image_init, center, visualize=False):
    # parameters
    markerLength_CL = 0.076 # verify
    aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_6X6_250) # verify
    parameters = aruco.DetectorParameters_create()

    corners_CL, ids_CL, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict_CL, parameters=parameters) # corners_CL: mean pixel ids_CL: id info

    # for the first frame, it may contain nothing
    if ids_CL is None:
        return default_M_CL

    outs = []
    for idx, id in enumerate(ids_CL):
        rvec_CL, tvec_CL, _objPoints_CL = aruco.estimatePoseSingleMarkers(corners_CL[idx], markerLength_CL,
                                                                          cameraMatrix, distCoeffs)
        dst_CL, jacobian_CL = cv2.Rodrigues(rvec_CL)
        M_CL = np.zeros((4, 4))
        M_CL[:3, :3] = dst_CL
        M_CL[:3, 3] = tvec_CL
        M_CL[3, :] = np.array([0, 0, 0, 1])

        if visualize:
            # print('aruco is located at mean position (%d, %d)' %(mean_x ,mean_y))
            aruco.drawAxis(image_init, cameraMatrix, distCoeffs, rvec_CL, tvec_CL, markerLength_CL)

        diagonal_pixel = None
        if id == 3:
            q_a = M_a3O.dot(center)
            q_c = M_CL.dot(q_a)
            p = cameraMatrix.dot(q_c[:-1])
            diagonal_pixel = p[:-1]
            diagonal_pixel /= p[-1]
        elif id == 4:
            q_a = M_a4O.dot(center)
            q_c = M_CL.dot(q_a)
            p = cameraMatrix.dot(q_c[:-1])
            diagonal_pixel = p[:-1]
            diagonal_pixel /= p[-1]
        elif id == 5:
            q_a = M_a5O.dot(center)
            q_c = M_CL.dot(q_a)
            p = cameraMatrix.dot(q_c[:-1])
            diagonal_pixel = p[:-1]
            diagonal_pixel /= p[-1]
        elif id == 6:
            q_a = M_a6O.dot(center)
            q_c = M_CL.dot(q_a)
            p = cameraMatrix.dot(q_c[:-1])
            diagonal_pixel = p[:-1]
            diagonal_pixel /= p[-1]
        else:
            raise ValueError('detected aruco tag id incorrect!')

        out = {'id': id[0],
               'mean_pixel': [corners_CL[idx][0][:, 0].mean(), corners_CL[idx][0][:, 1].mean()],
               'diagonal_pixel': diagonal_pixel,
               'M_CL': M_CL}
        outs.append(out)

    # organize detections according to ids
    sorted_outs = []
    for id in [3, 4, 5, 6]:
        for out in outs:
            if out['id'] == id:
                sorted_outs.append(out)
                break
    if visualize:
        cv2.namedWindow('image with aruco tag', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('image with aruco tag', image_init)
        cv2.waitKey(0)

    return sorted_outs
