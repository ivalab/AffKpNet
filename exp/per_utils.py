import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco

#####################################################
# Function: get the T matrix from camera to aruco tag
#####################################################
def get_M_CL(gray, image_init, visualize=False):
    # parameters
    markerLength_CL = 0.076
    # aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_5X5_250)
    aruco_dict_CL = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    corners_CL, ids_CL, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict_CL, parameters=parameters)

    # for the first frame, it may contain nothing
    if ids_CL is None:
        return default_M_CL

    rvec_CL, tvec_CL, _objPoints_CL = aruco.estimatePoseSingleMarkers(corners_CL[0], markerLength_CL,
                                                                      cameraMatrix, distCoeffs)
    dst_CL, jacobian_CL = cv2.Rodrigues(rvec_CL)
    M_CL = np.zeros((4, 4))
    M_CL[:3, :3] = dst_CL
    M_CL[:3, 3] = tvec_CL
    M_CL[3, :] = np.array([0, 0, 0, 1])

    print(M_CL)

    if visualize:
        # print('aruco is located at mean position (%d, %d)' %(mean_x ,mean_y))
        aruco.drawAxis(image_init, cameraMatrix, distCoeffs, rvec_CL, tvec_CL, markerLength_CL)
    return M_CL

def project(pixel, depth_image, M_CL, M_BL, cameraMatrix):
    '''
     project 2d pixel on the image to 3d by depth info
     :param pixel: x, y
     :param M_CL: trans from camera to aruco tag
     :param cameraMatrix: camera intrinsic matrix
     :param depth_image: depth image
     :param depth_scale: depth scale that trans raw data to mm
     :return:
     q_B: 3d coordinate of pixel with respect to base frame
     '''
    depth = depth_image[pixel[1], pixel[0]] * 0.001
    moving_pixel = [pixel[0], pixel[1]]
    while depth == 0:
        moving_pixel = [moving_pixel[0], moving_pixel[1]+1]
        depth = depth_image[moving_pixel[1], moving_pixel[0]] * 0.001

    pxl = np.linalg.inv(cameraMatrix).dot(
        np.array([pixel[0] * depth, pixel[1] * depth, depth]))
    q_C = np.array([pxl[0], pxl[1], pxl[2], 1])
    q_L = np.linalg.inv(M_CL).dot(q_C)
    q_B = M_BL.dot(q_L)

    return q_B

def compute_radius_wgrasp(mask, depth_image, M_CL, M_BL, cameraMatrix):
    z_axis_max = 0.
    idx_y, idx_x = np.where(mask == 6)
    for idx in range(len(idx_y)):
        if depth_image[idx_y[idx], idx_x[idx]] != 0:
            pxl = project([idx_x[idx], idx_y[idx]], depth_image, M_CL, M_BL, cameraMatrix)
            if pxl[2] > z_axis_max:
                z_axis_max = pxl[2]

    # average the depth of the found plane
    pxl_plane =[]
    idx_y, idx_x = np.where(mask == 4)
    for idx in range(len(idx_y)):
        pxl = project([idx_x[idx], idx_y[idx]], depth_image, M_CL, M_BL,
                      cameraMatrix)
        if np.abs(pxl[2] - z_axis_max) < 0.01:
            pxl_plane.append(pxl)

    dst_max = 0.
    for idx1 in range(len(pxl_plane)):
        for idx11 in range(idx1+1, len(pxl_plane)):
            dst = np.sqrt((pxl_plane[idx1][0] - pxl_plane[idx11][0])**2 + (pxl_plane[idx1][1] - pxl_plane[idx11][1])**2)
            if dst > dst_max:
                dst_max = dst

    return dst_max / 2.