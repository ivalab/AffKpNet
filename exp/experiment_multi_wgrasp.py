import sys, os
sys.path.append('/home/fujenchu/projects/affordanceContext/DANet/')

import os
import numpy as np
import cv2
from scipy import ndimage
import pyrealsense2 as rs

import torch
from torch import nn
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, BatchNorm2d
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from per_utils import get_M_CL, project, compute_radius_wgrasp
from kp_alg_utils import kp_post_process, class_vote_n_incorrect_fix, kp_nms_per_group, kp_nms_groups, \
                         geometric_contrain_filter, single_aff_filter

from option import Options
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

# aff color dict
color_dict = {1: [0,0,205], #grasp red
              2: [34,139,34], #cut green
              3: [0,255,255], #scoop bluegreen
              4: [165,42,42], #contain dark blue
              5: [128,64,128], #pound purple
              6: [184,134,11],#wrap-grasp light blue
              7: [0, 0, 0],}  #bg black

aff_dict = {'grasp': 1,
            'cut': 2,
            'scoop': 3,
            'contain': 4,
            'pound': 5,
            'wgrasp': 6}

M_CL = np.array([[-0.23821,    -0.96991648,  0.05017976, -0.16822528],
                 [-0.74092675,  0.14808033, -0.65505707,  0.13864789],
                 [ 0.62792002, -0.19322067, -0.75391128,  0.65720137],
                 [ 0.,          0.,          0.,          1.        ]]
                )

M_BL = np.array([[1., 0., 0.,  0.30000],
                 [0., 1., 0.,  0.32000],
                 [0., 0., 1.,  -0.0450],
                 [0., 0., 0.,  1.00000]])

cameraMatrix = np.array([[607.47165, 0.0,  325.90064],
                         [0.0, 606.30420, 240.91934],
                         [0.0, 0.0, 1.0]])
distCoeffs = np.array([0.08847, -0.04283, 0.00134, -0.00102, 0.0])

num_ouput = 500

def test(args, pipeline, align, depth_scale):
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux=args.aux,
                                       se_loss=args.se_loss, norm_layer=BatchNorm2d,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=args.multi_grid, multi_dilation=args.multi_dilation)

        if args.resume_dir is None or not os.path.isdir(args.resume_dir):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))

        for resume_file in os.listdir(args.resume_dir):
            if os.path.splitext(resume_file)[1] == '.tar':
                args.resume = os.path.join(args.resume_dir, resume_file)
        assert os.path.exists(args.resume)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    print(model)

    model.cuda()
    model.eval()

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            # Convert images to numpy arrays
            depth_raw = np.array(depth_frame.get_data()) * depth_scale
            depth = (depth_raw / depth_scale).astype(np.uint8)
            img = np.array(color_frame.get_data())
            gray = img.astype(np.uint8)

            # read the reference frame of the aruco tag
            M_CL = get_M_CL(gray, img, True)

            img_ori = img.copy()

            depth_ori = depth.copy()
            depth[depth > 1887] = 1887
            depth = (depth - 493) * 254 / (1887 - 493)
            depth[depth < 0] = 0

            # replace blue channel with the depth channel
            img[:, :, 2] = depth

            # normalization
            img = img / 255.
            img[:, :, 0] = (img[:, :, 0] - .358388) / .153398
            img[:, :, 1] = (img[:, :, 1] - .348858) / .137741
            img[:, :, 2] = (img[:, :, 2] - .294015) / .230031

            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            img_inp = img.copy()
            img = torch.from_numpy(img_inp)
            img = img.cuda()

            outputs = model(img.float())

            # affordance segmentation
            output_np = np.asarray(outputs['sasc'].data.cpu())
            output_np = output_np[0].transpose(1, 2, 0)
            output_np = np.asarray(np.argmax(output_np, axis=2), dtype=np.uint8)

            idx_y, idx_x = np.where(output_np != 0)
            mask = np.zeros((output_np.shape[0], output_np.shape[1], 3), dtype=np.uint8)
            for idx11 in range(len(idx_y)):
                mask[idx_y[idx11], idx_x[idx11], :] = color_dict[output_np[idx_y[idx11], idx_x[idx11]]]
            cv2.namedWindow('mask', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('mask', mask)


            # find the bound for each instance segmentation
            bounds = []
            binary_mask = output_np.copy()
            binary_mask[binary_mask != 0] = 1
            label_im, nb_labels = ndimage.label(binary_mask)
            for i in range(nb_labels):
                mask_compare = np.full(np.shape(label_im), i + 1)
                separate_mask = np.equal(label_im, mask_compare).astype(int)
                idx_y, idx_x = np.where(separate_mask == 1)
                min_x = np.min(idx_x)
                max_x = np.max(idx_x)
                min_y = np.min(idx_y)
                max_y = np.max(idx_y)
                bounds.append([min_x, max_x, min_y, max_y])

            for idx_s, bound in enumerate(bounds):
                min_x, max_x, min_y, max_y = bound[:]
                # give some tolerances
                min_x = min_x - 25
                max_x = max_x + 25
                min_y = min_y - 25
                max_y = max_y + 25

                outputs = model(img[:, :, min_y:max_y, min_x:max_x].float())

                # affordance segmentation
                output_np = np.asarray(outputs['sasc'].data.cpu())
                output_np = output_np[0].transpose(1, 2, 0)
                output_np = np.asarray(np.argmax(output_np, axis=2), dtype=np.uint8)

                idx_y, idx_x = np.where(output_np != 0)
                mask = np.zeros((output_np.shape[0], output_np.shape[1], 3), dtype=np.uint8)
                for idx11 in range(len(idx_y)):
                    mask[idx_y[idx11], idx_x[idx11], :] = color_dict[output_np[idx_y[idx11], idx_x[idx11]]]
                cv2.namedWindow('original image', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('original image', img_ori[min_y:max_y, min_x:max_x, :])
                cv2.namedWindow('mask', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('mask', mask)

                pred_kps = \
                kp_post_process(outputs, max_y-min_y, max_x-min_x, kernel=1, ae_threshold=1.0, scores_thresh=0.05, max_num_aff=50,
                                num_ouput=num_ouput)[0]

                # class vote and fix keypoints with incorrect class
                kps = class_vote_n_incorrect_fix(pred_kps)

                # remove keypoint highly overlapped results
                kps_f1 = kp_nms_per_group(kps)

                # remove highly overlapped grouping results
                kps_f2 = kp_nms_groups(kps_f1)

                kps_f3 = geometric_contrain_filter(kps_f2)

                # single output per aff
                kps_final = single_aff_filter(kps_f3)

                p_c_3d = None
                radius = 0.
                height = 0.
                for kp in kps_final:
                    if kp[11] == 5:
                        # project operating location to 3d
                        p_c = [int(kp[8] + min_x), int(kp[9] + min_y)]
                        p_c_3d = project(p_c, depth_ori, M_CL, M_BL, cameraMatrix)

                        # compute the radius of cylinder
                        radius = compute_radius_wgrasp(output_np, depth_ori[min_y:max_y, min_x:max_x], M_CL, M_BL, cameraMatrix)

                        # compute the height of the cylinder
                        p_3 = [int(kp[4] + min_x), int(kp[5] + min_y)]
                        p_3_3d = project(p_3, depth_ori, M_CL, M_BL, cameraMatrix)
                        while depth_ori[p_3[1], p_3[0]] == 0:
                            p_3[1] = p_3[1] - 1
                            p_3_3d = project(p_3, depth_ori, M_CL, M_BL, cameraMatrix)

                        height = p_3_3d[2] + 0.045

                        img_show = img_ori.copy()
                        img_show = cv2.circle(img_show, (int(kp[0] + min_x), int(kp[1] + min_y)), 2, (0, 0, 255), 2)  # red
                        img_show = cv2.circle(img_show, (int(kp[2] + min_x), int(kp[3] + min_y)), 2, (0, 255, 0), 2)  # green
                        img_show = cv2.circle(img_show, (int(kp[4] + min_x), int(kp[5] + min_y)), 2, (255, 0, 0), 2)  # blue
                        img_show = cv2.circle(img_show, (int(kp[6] + min_x), int(kp[7] + min_y)), 2, (220, 0, 255), 2)  # purple pink
                        img_show = cv2.circle(img_show, (int(kp[8] + min_x), int(kp[9] + min_y)), 2, (0, 0, 0), 2)  # dark
                        img_show = cv2.putText(img_show, 'Aff id:' + str(kp[11]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                               (255, 0, 0), 2, cv2.LINE_AA)

                print('cylinder radius: {}'.format(radius))
                print('cylinder height: {}'.format(height))
                p_c_3d[0] += (radius * 2)
                print('operating location: {}'.format(p_c_3d))

                cv2.namedWindow('visual', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('visual', img_show)
                cv2.waitKey(0)
    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    args = Options().parse()

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    align_to = rs.stream.color
    align = rs.align(align_to)

    test(args, pipeline, align, depth_scale)