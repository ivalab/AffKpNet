import sys
import math
import cv2
import numpy as np
import os
import scipy.io as sio
import shutil

# all files will be moved to folder of tools_correctized
SAVEDIR = '../../images/complementary/self_collected_correct'
if not os.path.exists(SAVEDIR):
    os.mkdir(SAVEDIR)
# folder path of original datasetdd
ROOTDIR = '../../images/complementary/self_collected'
# object name, need to be changed every time
OBJECT = 'spoon_05'
# ROOTDIR = os.path.join(ROOTDIR, 'annotations_kp')
if not os.path.exists(os.path.join(SAVEDIR, 'annotations_kp')):
    os.mkdir(os.path.join(SAVEDIR, 'annotations_kp'))
SAVEDIR = os.path.join(SAVEDIR, 'annotations_kp')
if not os.path.exists(os.path.join(SAVEDIR, OBJECT)):
    os.mkdir(os.path.join(SAVEDIR, OBJECT))

NM_IMG = 360
START_IDX = 351

refPt = []
img_cor = np.empty((480, 640, 3))
def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt
    # if the left mouse button was clicked, record the location of clicked pixel
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > 639 or x < 0 or y > 479 or y < 0:
            print("Out of range")
        else:
            refPt.append([x, y])

            global img_cor
            img_cor = cv2.circle(img_cor, (int(x), int(y)), 2, (0, 0, 255), -1)
            cv2.putText(img_cor, 'correctized', (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("visual correct", img_cor)

for idx in range(START_IDX, NM_IMG):
    print('\n')
    print("Right now you are check no.{} image".format(idx))
    img = cv2.imread(os.path.join(ROOTDIR, 'images', OBJECT, 'rgb_' + str(idx) + '.png'))

    file_kp = open(os.path.join(ROOTDIR, 'annotations_kp', OBJECT, str(idx) + '_keypoint.txt'), "r")

    img_show = img.copy()

    file_kp_saved = open(os.path.join(SAVEDIR, OBJECT, str(idx) + '_keypoint.txt'), "w")

    affordance = file_kp.readline()
    while affordance:
        # if affordance.split()[0] == 'grasp':
        print('You are annotating {} affordance now'.format(affordance.split('\n')[0]))
        line1 = file_kp.readline().split()
        p1 = list(map(int, [round(float(line1[0])), round(float(line1[1]))]))
        line2 = file_kp.readline().split()
        p2 = list(map(int, [round(float(line2[0])), round(float(line2[1]))]))
        line3 = file_kp.readline().split()
        p3 = list(map(int, [round(float(line3[0])), round(float(line3[1]))]))
        line4 = file_kp.readline().split()
        p4 = list(map(int, [round(float(line4[0])), round(float(line4[1]))]))
        line5 = file_kp.readline().split()
        p5 = list(map(int, [round(float(line5[0])), round(float(line5[1]))]))
        points = np.array([p1, p2, p3, p4, p5])

        for idx_p, p in enumerate(points):
            #img_show = cv2.circle(img_show, (p[0], p[1]), 2, (0, 0, 255), -1)
            if idx_p == 0:
                cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
            elif idx_p == 1:
                cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
            elif idx_p == 2:
                cv2.putText(img_show, str(idx_p + 1), (p[0] - 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
            elif idx_p == 3:
                cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
            elif idx_p == 4:
                cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), 1, cv2.LINE_AA)
        for idx_p, p in enumerate(points):
            img_show = cv2.circle(img_show, (p[0], p[1]), 2, (0, 0, 255), -1)

        cv2.namedWindow('visual', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('visual', img_show)
        print('Press certain keypoint to shift to that direction by one pixel, otherwise press any other keys to finish checking\n'
              '           8           \n'
              '           ^           \n'
              '           |           \n'
              '           |           \n'
              '4   <--         -->   6\n'
              '           |           \n'
              '           |           \n'
              '           v           \n'
              '           2           ')
        k = cv2.waitKey(0)

        while k == ord('2') or k == ord('4') or k == ord('6') or k == ord('8'):
            img_show = img.copy()
            if k == ord('2'):
                for idx_p, p in enumerate(points):
                    # img_show = cv2.circle(img_show, (p[0], p[1]), 2, (0, 0, 255), -1)
                    if idx_p == 0:
                        cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 1:
                        cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 2:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] - 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 3:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 4:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                for idx_p, p in enumerate(points):
                    points[idx_p] = [p[0], p[1] + 1]
                    img_show = cv2.circle(img_show, (p[0], p[1] + 1), 2, (0, 0, 255), -1)
            elif k == ord('8'):
                for idx_p, p in enumerate(points):
                    # img_show = cv2.circle(img_show, (p[0], p[1]), 2, (0, 0, 255), -1)
                    if idx_p == 0:
                        cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 1:
                        cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 2:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] - 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 3:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 4:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                for idx_p, p in enumerate(points):
                    points[idx_p] = [p[0], p[1] - 1]
                    img_show = cv2.circle(img_show, (p[0], p[1] - 1), 2, (0, 0, 255), -1)
            elif k == ord('4'):
                for idx_p, p in enumerate(points):
                    # img_show = cv2.circle(img_show, (p[0], p[1]), 2, (0, 0, 255), -1)
                    if idx_p == 0:
                        cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 1:
                        cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 2:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] - 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 3:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 4:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                for idx_p, p in enumerate(points):
                    points[idx_p] = [p[0] - 1, p[1]]
                    img_show = cv2.circle(img_show, (p[0] - 1, p[1]), 2, (0, 0, 255), -1)
            elif k == ord('6'):
                for idx_p, p in enumerate(points):
                    # img_show = cv2.circle(img_show, (p[0], p[1]), 2, (0, 0, 255), -1)
                    if idx_p == 0:
                        cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 1:
                        cv2.putText(img_show, str(idx_p + 1), (p[0], p[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 2:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] - 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 3:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    elif idx_p == 4:
                        cv2.putText(img_show, str(idx_p + 1), (p[0] + 5, p[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.2,
                                    (255, 255, 255), 1, cv2.LINE_AA)
                for idx_p, p in enumerate(points):
                    points[idx_p] = [p[0] + 1, p[1]]
                    img_show = cv2.circle(img_show, (p[0] + 1, p[1]), 2, (0, 0, 255), -1)

            cv2.namedWindow('visual', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('visual', img_show)
            k = cv2.waitKey(0)

        file_kp_saved.write(affordance)
        file_kp_saved.write(str(points[0][0]) + ' ' + str(points[0][1]) + '\n')
        file_kp_saved.write(str(points[1][0]) + ' ' + str(points[1][1]) + '\n')
        file_kp_saved.write(str(points[2][0]) + ' ' + str(points[2][1]) + '\n')
        file_kp_saved.write(str(points[3][0]) + ' ' + str(points[3][1]) + '\n')
        file_kp_saved.write(str(points[4][0]) + ' ' + str(points[4][1]) + '\n')

        affordance = file_kp.readline()
        # clear annotated image
        img_show = img.copy()

    file_kp.close()
    file_kp_saved.close()




