###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn 
# Copyright (c) 2018
###########################################################################
import sys, os
sys.path.append('/home/fujenchu/projects/affordanceContext/DANet/')

import os
import numpy as np
from tqdm import tqdm
import cv2
import scipy.io as sio

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

from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi

from option import Options
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

GT_ROOT = '../datasets/UMD_affordance/compound'

# aff color dict
color_dict = {1: [0, 0, 205], #grasp red
              2: [34, 139, 34], #cut green
              3: [0, 255, 255], #scoop bluegreen
              4: [165, 42, 42], #contain dark blue
              5: [128, 64, 128], #pound purple
              6: [184, 134, 11],#wrap-grasp light blue
              7: [0, 0, 0],}  #bg black

aff_dict = {'grasp': 1,
            'cut': 2,
            'scoop': 3,
            'contain': 4,
            'pound': 5,
            'wgrasp': 6}

# dict for aff id with normalization factor
d_dict = {1: 32.26,
          2: 28.08,
          3: 19.40,
          4: 32.55,
          5: 25.25,
          6: 42.01}

def test(args):
    # output folder
    outdir = '%s/danet_vis'%(args.dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.358388, .348858, .294015], [.153398, .137741, .230031])])

    # dataset
    if args.eval:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval',
                                           transform=input_transform)
    elif args.eval_kp:
        testset = get_segmentation_dataset(args.dataset, split='val', mode='testval_kp',
                                           transform=input_transform)
    else:#set split='test' for test set
        testset = get_segmentation_dataset(args.dataset, split='val', mode='vis',
                                           transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False, **loader_kwargs)
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux=args.aux,
                                       se_loss=args.se_loss, norm_layer=BatchNorm2d,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=args.multi_grid, multi_dilation=args.multi_dilation)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    print(model)

    model.cuda()
    model.eval()

    tbar = tqdm(test_data)

    def eval_batch(image_inp, dst, img_path, model, nme_dict, nme_count_dict, p_dict, r_dict):
        # evaluation mode on validation set
        outputs = model(image_inp)

        object_name = img_path[0].split('/')[-2]

        # load gt keypoints as dict
        kp_dict = {}
        kp_name = img_path[0].split('/')[-1].replace('labelid.png', 'keypoint.txt')
        file = open(os.path.join(GT_ROOT, object_name, kp_name), 'r')
        affordance = file.readline()
        while affordance:
            line1 = file.readline().split()
            line2 = file.readline().split()
            line3 = file.readline().split()
            line4 = file.readline().split()
            line5 = file.readline().split()
            kp_dict[aff_dict[affordance.split('\n')[0]]] = [float(line1[0]), float(line1[1]), \
                                                            float(line2[0]), float(line2[1]), \
                                                            float(line3[0]), float(line3[1]), \
                                                            float(line4[0]), float(line4[1]), \
                                                            float(line5[0]), float(line5[1])]
            affordance = file.readline()
        file.close()

        sym_flag = {}
        flag_file_name = kp_name.replace('keypoint.txt', 'func_sym_flag.txt')
        file_flag = open(os.path.join(GT_ROOT, object_name, flag_file_name), 'r')
        line = file_flag.readline()
        while line:
            line = line.split()
            sym_flag[aff_dict[line[0]]] = int(line[1])
            line = file_flag.readline()
        file_flag.close()

        num_ouput = 500
        pred_kps = kp_post_process(outputs, 480, 640, max_num_aff=50, num_ouput=num_ouput)[0]

        # class vote and fix keypoints with incorrect class
        kps = class_vote_n_incorrect_fix(pred_kps)

        # nms per group and groups
        kps_f = kp_nms(kps)

        # single output per aff
        kps_final = single_aff_filter(kps_f)

        # compute NME
        for aff_id, gt_kps in kp_dict.items():
            if aff_id - 1 in kps_final.keys():
                pred_kp = kps_final[aff_id - 1]
                v_nme = sym_nme(pred_kp, gt_kps, d_dict[aff_id], sym_flag[aff_id])
                nme_dict[aff_id] += v_nme
                nme_count_dict[aff_id] += 1

                if v_nme <= 0.1 * d_dict[aff_id]:
                    p_dict[aff_id] += 1
                    r_dict[aff_id] += 1
                else:
                    r_dict[aff_id] += 1
            else:
                r_dict[aff_id] += 1
                continue

    nme_dict = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}

    nme_count_dict = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}

    p_dict = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}

    r_dict = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}

    for i, (image_inp, dst, img_path) in enumerate(tbar):
        with torch.no_grad():
            image_inp = image_inp.cuda()
            for key, value in dst.items():
                if not dst[key].is_cuda:
                    dst[key] = value.cuda()
            eval_batch(image_inp, dst, img_path, model, nme_dict, nme_count_dict, p_dict, r_dict)


    print(nme_dict)
    print(nme_count_dict)
    print(p_dict)
    print(r_dict)


def eval_multi_models(args):
    if args.resume_dir is None or not os.path.isdir(args.resume_dir):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))
    for resume_file in os.listdir(args.resume_dir):
        if os.path.splitext(resume_file)[1] == '.tar':
            args.resume = os.path.join(args.resume_dir, resume_file)
            assert os.path.exists(args.resume)
            if not args.eval:
                test(args)
                continue
            test(args)

    print('Evaluation is finished!!!')

def sym_nme(pred_kp, gt_kp, d, flag):
    v1 = 0.
    if flag == 0:
        for idx in range(5):
            v1 += np.sqrt(
                ((pred_kp[2 * idx] - gt_kp[2 * idx]) ** 2 + (pred_kp[2 * idx + 1] - gt_kp[2 * idx + 1]) ** 2)) / d
    elif flag == 1:
        d1 = np.sqrt(((pred_kp[2 * 0] - gt_kp[2 * 0]) ** 2 + (pred_kp[2 * 0 + 1] - gt_kp[2 * 0 + 1]) ** 2)) / d + \
             np.sqrt(((pred_kp[2 * 1] - gt_kp[2 * 1]) ** 2 + (pred_kp[2 * 1 + 1] - gt_kp[2 * 1 + 1]) ** 2)) / d
        d2 = np.sqrt(((pred_kp[2 * 0] - gt_kp[2 * 1]) ** 2 + (pred_kp[2 * 0 + 1] - gt_kp[2 * 1 + 1]) ** 2)) / d + \
             np.sqrt(((pred_kp[2 * 1] - gt_kp[2 * 0]) ** 2 + (pred_kp[2 * 1 + 1] - gt_kp[2 * 0 + 1]) ** 2)) / d

        np.minimum(d1, d2)
        for idx in range(2, 5):
            v1 += np.sqrt(
                ((pred_kp[2 * idx] - gt_kp[2 * idx]) ** 2 + (pred_kp[2 * idx + 1] - gt_kp[2 * idx + 1]) ** 2)) / d

    elif flag == 2:
        d1 = np.sqrt(((pred_kp[2 * 2] - gt_kp[2 * 2]) ** 2 + (pred_kp[2 * 2 + 1] - gt_kp[2 * 2 + 1]) ** 2)) / d + \
             np.sqrt(((pred_kp[2 * 3] - gt_kp[2 * 3]) ** 2 + (pred_kp[2 * 3 + 1] - gt_kp[2 * 3 + 1]) ** 2)) / d
        d2 = np.sqrt(((pred_kp[2 * 2] - gt_kp[2 * 3]) ** 2 + (pred_kp[2 * 2 + 1] - gt_kp[2 * 3 + 1]) ** 2)) / d + \
             np.sqrt(((pred_kp[2 * 3] - gt_kp[2 * 2]) ** 2 + (pred_kp[2 * 3 + 1] - gt_kp[2 * 2 + 1]) ** 2)) / d

        v1 += np.minimum(d1, d2)
        v1 += np.sqrt(((pred_kp[2 * 0] - gt_kp[2 * 0]) ** 2 + (pred_kp[2 * 0 + 1] - gt_kp[2 * 0 + 1]) ** 2)) / d + \
              np.sqrt(((pred_kp[2 * 1] - gt_kp[2 * 1]) ** 2 + (pred_kp[2 * 1 + 1] - gt_kp[2 * 1 + 1]) ** 2)) / d + \
              np.sqrt(((pred_kp[2 * 4] - gt_kp[2 * 4]) ** 2 + (pred_kp[2 * 4 + 1] - gt_kp[2 * 4 + 1]) ** 2)) / d

    elif flag == 3:
        d1 = np.sqrt(((pred_kp[2 * 0] - gt_kp[2 * 0]) ** 2 + (pred_kp[2 * 0 + 1] - gt_kp[2 * 0 + 1]) ** 2)) / d + \
             np.sqrt(((pred_kp[2 * 1] - gt_kp[2 * 1]) ** 2 + (pred_kp[2 * 1 + 1] - gt_kp[2 * 1 + 1]) ** 2)) / d
        d2 = np.sqrt(((pred_kp[2 * 0] - gt_kp[2 * 1]) ** 2 + (pred_kp[2 * 0 + 1] - gt_kp[2 * 1 + 1]) ** 2)) / d + \
             np.sqrt(((pred_kp[2 * 1] - gt_kp[2 * 0]) ** 2 + (pred_kp[2 * 1 + 1] - gt_kp[2 * 0 + 1]) ** 2)) / d
        d3 = np.sqrt(((pred_kp[2 * 2] - gt_kp[2 * 2]) ** 2 + (pred_kp[2 * 2 + 1] - gt_kp[2 * 2 + 1]) ** 2)) / d + \
             np.sqrt(((pred_kp[2 * 3] - gt_kp[2 * 3]) ** 2 + (pred_kp[2 * 3 + 1] - gt_kp[2 * 3 + 1]) ** 2)) / d
        d4 = np.sqrt(((pred_kp[2 * 2] - gt_kp[2 * 3]) ** 2 + (pred_kp[2 * 2 + 1] - gt_kp[2 * 3 + 1]) ** 2)) / d + \
             np.sqrt(((pred_kp[2 * 3] - gt_kp[2 * 2]) ** 2 + (pred_kp[2 * 3 + 1] - gt_kp[2 * 2 + 1]) ** 2)) / d
        v1 += np.minimum(d1, d2)
        v1 += np.minimum(d3, d4)
        v1 += np.sqrt(((pred_kp[2 * 4] - gt_kp[2 * 4]) ** 2 + (pred_kp[2 * 4 + 1] - gt_kp[2 * 4 + 1]) ** 2)) / d
    return v1

def kp_nms(kps_in):
    kps_iter = []
    for kp in kps_in:
        flag = True

        for idx1 in range(5):
            for idx11 in range(idx1 + 1, 5):
                if (idx1 == 3 and idx11 == 5) or (idx1 == 4 and idx11 == 5):
                    continue
                if np.sqrt(np.sum((kp[idx1 * 2:(idx1 + 1) * 2] - kp[idx11 * 2:(idx11 + 1) * 2]) ** 2)) < 4 * np.sqrt(2):
                    flag = False
        if flag:
            kps_iter.append(kp)

    # return input else all detections are highly cluttered else return filtered one
    if len(kps_out) == 0:
        kps_iter = kps_in
    else:
        kps_iter = np.array(kps_out)

    kps_out = np.expand_dims(kps_iter[0, :], axis=0)  # always select the first one
    for kp in kps_iter:
        flag = True
        for idx111 in range(kps_out.shape[0]):
            if np.sqrt(np.sum((kps_out[idx111, :10] - np.expand_dims(kp[:10], axis=0)) ** 2)) < 40:
                flag = False

        if flag:
            kps_out = np.concatenate((kps_out, np.expand_dims(kp, axis=0)), axis=0)

    return kps_out


def class_vote_n_incorrect_fix(pred_kps):
    kps = []
    for pred_idx in range(pred_kps.shape[0]):
        pred_kp = pred_kps[pred_idx, :]
        cls = np.bincount(pred_kp[11:16].astype(np.int64)).argmax()
        # if pred_kp[10] > 0.1:
        if cls == 5:
            ###########################
            # incorrect labeled kps fix
            ###########################
            # Case 1: three keypoints should have the same label
            if np.where(pred_kp[11:16] == cls)[0].size == 3 and pred_kp[15] == cls:
                p5 = [pred_kp[8], pred_kp[9]]
                # no.1,2 or 3,4 can't be the same, otherwise has no reference to fix the missing one
                if pred_kp[11] == pred_kp[12] or pred_kp[13] == pred_kp[14]:
                    continue
                if pred_kp[11] == cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p2 = [2 * p5[0] - p1[0], p1[1]]
                elif pred_kp[12] == cls:
                    p2 = [pred_kp[2], pred_kp[3]]
                    p1 = [2 * p5[0] - p2[0], p2[1]]
                if pred_kp[13] == cls:
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [2 * p5[0] - p3[0], 2 * p5[1] - p3[1]]
                elif pred_kp[14] == cls:
                    p4 = [pred_kp[6], pred_kp[7]]
                    p3 = [2 * p5[0] - p4[0], 2 * p5[1] - p4[1]]
                kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                kps.append(kp_new)

            elif np.where(pred_kp[11:16] == cls)[0].size == 3 and pred_kp[15] != cls:
                if pred_kp[11] == cls and pred_kp[12] == cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p2 = [pred_kp[2], pred_kp[3]]
                    p5 = [(p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.]
                    if pred_kp[13] == cls and pred_kp[14] != cls:
                        p3 = [pred_kp[4], pred_kp[5]]
                        p4 = [2 * p5[0] - p3[0], 2 * p5[1] - p3[1]]
                    elif pred_kp[13] != cls and pred_kp[14] == cls:
                        p4 = [pred_kp[6], pred_kp[7]]
                        p3 = [2 * p5[0] - p4[0], 2 * p5[1] - p4[1]]
                    kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                    kps.append(kp_new)
                elif pred_kp[13] == cls and pred_kp[14] == cls:
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [pred_kp[6], pred_kp[7]]
                    p5 = [(p3[0] + p4[0]) / 2., (p3[1] + p4[1]) / 2.]
                    if pred_kp[11] == cls and pred_kp[12] != cls:
                        p1 = [pred_kp[0], pred_kp[1]]
                        p2 = [2 * p5[0] - p1[0], p1[1]]
                    elif pred_kp[11] != cls and pred_kp[12] == cls:
                        p2 = [pred_kp[2], pred_kp[3]]
                        p1 = [2 * p5[0] - p2[0], p2[1]]
                    kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                    kps.append(kp_new)

            # Case 2: four keypoints should have the same label
            if np.where(pred_kp[11:16] == cls)[0].size == 4 and pred_kp[15] != cls:
                p1 = [pred_kp[0], pred_kp[1]]
                p2 = [pred_kp[2], pred_kp[3]]
                p3 = [pred_kp[4], pred_kp[5]]
                p4 = [pred_kp[6], pred_kp[7]]
                p5 = [(p1[0] + p2[0]) / 2., (p3[1] + p4[1]) / 2.]
                kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                kps.append(kp_new)
            elif np.where(pred_kp[11:16] == cls)[0].size == 4 and pred_kp[15] == cls:
                p5 = [pred_kp[8], pred_kp[9]]
                if pred_kp[11] != cls:
                    p2 = [pred_kp[2], pred_kp[3]]
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [pred_kp[6], pred_kp[7]]
                    p1 = [2 * p5[0] - p2[0], p2[1]]
                elif pred_kp[12] != cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [pred_kp[6], pred_kp[7]]
                    p2 = [2 * p5[0] - p1[0], p1[1]]
                elif pred_kp[13] != cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p2 = [pred_kp[2], pred_kp[3]]
                    p4 = [pred_kp[6], pred_kp[7]]
                    p3 = [2 * p5[0] - p4[0], 2 * p5[1] - p4[1]]
                elif pred_kp[14] != cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p2 = [pred_kp[2], pred_kp[3]]
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [2 * p5[0] - p3[0], 2 * p5[1] - p3[1]]
                kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                kps.append(kp_new)

            # Case 3: five keypoints should have the same label
            if np.where(pred_kp[11:16] == cls)[0].size == 5:
                p1 = [pred_kp[0], pred_kp[1]]
                p2 = [pred_kp[2], pred_kp[3]]
                p3 = [pred_kp[4], pred_kp[5]]
                p4 = [pred_kp[6], pred_kp[7]]
                p5 = [pred_kp[8], pred_kp[9]]
                kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                kps.append(kp_new)
        else:
            ###########################
            # incorrect labeled kps fix
            ###########################
            # Case 1: three keypoints should have the same label
            if np.where(pred_kp[11:16] == cls)[0].size == 3 and pred_kp[15] == cls:
                p5 = [pred_kp[8], pred_kp[9]]
                # no.1,2 or 3,4 can't be the same, otherwise has no reference to fix the missing one
                if pred_kp[11] == pred_kp[12] or pred_kp[13] == pred_kp[14]:
                    continue
                if pred_kp[11] == cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p2 = [2 * p5[0] - p1[0], 2 * p5[1] - p1[1]]
                elif pred_kp[12] == cls:
                    p2 = [pred_kp[2], pred_kp[3]]
                    p1 = [2 * p5[0] - p2[0], 2 * p5[1] - p2[1]]
                if pred_kp[13] == cls:
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [2 * p5[0] - p3[0], 2 * p5[1] - p3[1]]
                elif pred_kp[14] == cls:
                    p4 = [pred_kp[6], pred_kp[7]]
                    p3 = [2 * p5[0] - p4[0], 2 * p5[1] - p4[1]]
                kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                kps.append(kp_new)

            elif np.where(pred_kp[11:16] == cls)[0].size == 3 and pred_kp[15] != cls:
                if pred_kp[11] == cls and pred_kp[12] == cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p2 = [pred_kp[2], pred_kp[3]]
                    p5 = [(p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.]
                    if pred_kp[13] == cls and pred_kp[14] != cls:
                        p3 = [pred_kp[4], pred_kp[5]]
                        p4 = [2 * p5[0] - p3[0], 2 * p5[1] - p3[1]]
                    elif pred_kp[13] != cls and pred_kp[14] == cls:
                        p4 = [pred_kp[6], pred_kp[7]]
                        p3 = [2 * p5[0] - p4[0], 2 * p5[1] - p4[1]]
                    kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                    kps.append(kp_new)
                elif pred_kp[13] == cls and pred_kp[14] == cls:
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [pred_kp[6], pred_kp[7]]
                    p5 = [(p3[0] + p4[0]) / 2., (p3[1] + p4[1]) / 2.]
                    if pred_kp[11] == cls and pred_kp[12] != cls:
                        p1 = [pred_kp[0], pred_kp[1]]
                        p2 = [2 * p5[0] - p1[0], 2 * p5[1] - p1[1]]
                    elif pred_kp[11] != cls and pred_kp[12] == cls:
                        p2 = [pred_kp[2], pred_kp[3]]
                        p1 = [2 * p5[0] - p2[0], 2 * p5[1] - p2[1]]
                    kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                    kps.append(kp_new)

            # Case 2: four keypoints should have the same label
            if np.where(pred_kp[11:16] == cls)[0].size == 4 and pred_kp[15] != cls:
                p1 = [pred_kp[0], pred_kp[1]]
                p2 = [pred_kp[2], pred_kp[3]]
                p3 = [pred_kp[4], pred_kp[5]]
                p4 = [pred_kp[6], pred_kp[7]]
                p5 = [(p1[0] + p2[0]) / 2., (p3[1] + p4[1]) / 2.]
                kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                kps.append(kp_new)
            elif np.where(pred_kp[11:16] == cls)[0].size == 4 and pred_kp[15] == cls:
                p5 = [pred_kp[8], pred_kp[9]]
                if pred_kp[11] != cls:
                    p2 = [pred_kp[2], pred_kp[3]]
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [pred_kp[6], pred_kp[7]]
                    p1 = [2 * p5[0] - p2[0], 2 * p5[1] - p2[1]]
                elif pred_kp[12] != cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [pred_kp[6], pred_kp[7]]
                    p2 = [2 * p5[0] - p1[0], 2 * p5[1] - p1[1]]
                elif pred_kp[13] != cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p2 = [pred_kp[2], pred_kp[3]]
                    p4 = [pred_kp[6], pred_kp[7]]
                    p3 = [2 * p5[0] - p4[0], 2 * p5[1] - p4[1]]
                elif pred_kp[14] != cls:
                    p1 = [pred_kp[0], pred_kp[1]]
                    p2 = [pred_kp[2], pred_kp[3]]
                    p3 = [pred_kp[4], pred_kp[5]]
                    p4 = [2 * p5[0] - p3[0], 2 * p5[1] - p3[1]]
                kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                kps.append(kp_new)

            # Case 3: five keypoints should have the same label
            if np.where(pred_kp[11:16] == cls)[0].size == 5:
                p1 = [pred_kp[0], pred_kp[1]]
                p2 = [pred_kp[2], pred_kp[3]]
                p3 = [pred_kp[4], pred_kp[5]]
                p4 = [pred_kp[6], pred_kp[7]]
                p5 = [pred_kp[8], pred_kp[9]]
                kp_new = p1 + p2 + p3 + p4 + p5 + [pred_kp[10]] + [cls] + list(pred_kp[16:21])
                kps.append(kp_new)

    kps = np.array(kps)

    return kps

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)


    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def kp_post_process(output, input_h, input_w, max_num_aff=3, num_ouput=3):
    with torch.no_grad():
        batch, cat, height, width = output['1'].size()

        f_heatmap = output['1'].sigmoid_()
        s_heatmap = output['2'].sigmoid_()
        t_heatmap = output['3'].sigmoid_()
        fo_heatmap = output['4'].sigmoid_()
        fi_heatmap = output['5'].sigmoid_()

        f_tag = output['1_tag']
        s_tag = output['2_tag']
        t_tag = output['3_tag']
        fo_tag = output['4_tag']
        fi_tag = output['5_tag']

        f_reg = output['1_reg']
        s_reg = output['2_reg']
        t_reg = output['3_reg']
        fo_reg = output['4_reg']
        fi_reg = output['5_reg']

        f_heatmap[f_heatmap > 1] = 1
        s_heatmap[s_heatmap > 1] = 1
        t_heatmap[t_heatmap > 1] = 1
        fo_heatmap[fo_heatmap > 1] = 1
        fi_heatmap[fi_heatmap > 1] = 1

        f_scores, f_inds, f_clses, f_ys, f_xs = _topk(f_heatmap, K=max_num_aff)
        s_scores, s_inds, s_clses, s_ys, s_xs = _topk(s_heatmap, K=max_num_aff)
        t_scores, t_inds, t_clses, t_ys, t_xs = _topk(t_heatmap, K=max_num_aff)
        fo_scores, fo_inds, fo_clses, fo_ys, fo_xs = _topk(fo_heatmap, K=max_num_aff)
        fi_scores, fi_inds, fi_clses, fi_ys, fi_xs = _topk(fi_heatmap, K=max_num_aff)

        # filter highly overlapped in heatmap-level
        f_scores, f_inds, f_clses, f_ys, f_xs, K1 = filter_overlap_hl(f_scores, f_inds, f_clses, f_ys, f_xs)
        s_scores, s_inds, s_clses, s_ys, s_xs, K2 = filter_overlap_hl(s_scores, s_inds, s_clses, s_ys, s_xs)
        t_scores, t_inds, t_clses, t_ys, t_xs, K3 = filter_overlap_hl(t_scores, t_inds, t_clses, t_ys, t_xs)
        fo_scores, fo_inds, fo_clses, fo_ys, fo_xs, K4 = filter_overlap_hl(fo_scores, fo_inds, fo_clses, fo_ys, fo_xs)
        fi_scores, fi_inds, fi_clses, fi_ys, fi_xs, K5 = filter_overlap_hl(fi_scores, fi_inds, fi_clses, fi_ys, fi_xs)

        f_ys = f_ys.view(batch, K1, 1, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        f_xs = f_xs.view(batch, K1, 1, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        s_ys = s_ys.view(batch, 1, K2, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        s_xs = s_xs.view(batch, 1, K2, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        t_ys = t_ys.view(batch, 1, 1, K3, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        t_xs = t_xs.view(batch, 1, 1, K3, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        fo_ys = fo_ys.view(batch, 1, 1, 1, K4, 1).expand(batch, K1, K2, K3, K4, K5)
        fo_xs = fo_xs.view(batch, 1, 1, 1, K4, 1).expand(batch, K1, K2, K3, K4, K5)
        fi_ys = fi_ys.view(batch, 1, 1, 1, 1, K5).expand(batch, K1, K2, K3, K4, K5)
        fi_xs = fi_xs.view(batch, 1, 1, 1, 1, K5).expand(batch, K1, K2, K3, K4, K5)

        f_clses = f_clses.view(batch, K1, 1, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        s_clses = s_clses.view(batch, 1, K2, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        t_clses = t_clses.view(batch, 1, 1, K3, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        fo_clses = fo_clses.view(batch, 1, 1, 1, K4, 1).expand(batch, K1, K2, K3, K4, K5)
        fi_clses = fi_clses.view(batch, 1, 1, 1, 1, K5).expand(batch, K1, K2, K3, K4, K5)

        f_tag = _transpose_and_gather_feat(f_tag, f_inds)
        f_tag = f_tag.view(batch, K1, 1, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        s_tag = _transpose_and_gather_feat(s_tag, s_inds)
        s_tag = s_tag.view(batch, 1, K2, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        t_tag = _transpose_and_gather_feat(t_tag, t_inds)
        t_tag = t_tag.view(batch, 1, 1, K3, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        fo_tag = _transpose_and_gather_feat(fo_tag, fo_inds)
        fo_tag = fo_tag.view(batch, 1, 1, 1, K4, 1).expand(batch, K1, K2, K3, K4, K5)
        fi_tag = _transpose_and_gather_feat(fi_tag, fi_inds)
        fi_tag = fi_tag.view(batch, 1, 1, 1, 1, K5).expand(batch, K1, K2, K3, K4, K5)


        f_scores = f_scores.view(batch, K1, 1, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        s_scores = s_scores.view(batch, 1, K2, 1, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        t_scores = t_scores.view(batch, 1, 1, K3, 1, 1).expand(batch, K1, K2, K3, K4, K5)
        fo_scores = fo_scores.view(batch, 1, 1, 1, K4, 1).expand(batch, K1, K2, K3, K4, K5)
        fi_scores = fi_scores.view(batch, 1, 1, 1, 1, K5).expand(batch, K1, K2, K3, K4, K5)
        scores = (f_scores + s_scores + t_scores + fo_scores + fi_scores)

        average_aes = (f_tag + s_tag + t_tag + fo_tag + fi_tag) / 5.
        dists = torch.sqrt((f_tag - average_aes).pow(2) +
                           (s_tag - average_aes).pow(2) +
                           (t_tag - average_aes).pow(2) +
                           (fo_tag - average_aes).pow(2) +
                           (fi_tag - average_aes).pow(2))
        scores = scores - dists

        if K1 * K2 * K3 * K4 * K5 < num_ouput:
            num_ouput = K1 * K2 * K3 * K4 * K5
        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_ouput)  # max_num_aff
        scores = scores.unsqueeze(2)

        if f_reg is not None and s_reg is not None and t_reg is not None and fo_reg is not None and fi_reg is not None:
            f_reg = _transpose_and_gather_feat(f_reg, f_inds)
            f_reg = f_reg.view(batch, K1, 1, 1, 1, 1, 2)
            s_reg = _transpose_and_gather_feat(s_reg, s_inds)
            s_reg = s_reg.view(batch, 1, K2, 1, 1, 1, 2)
            t_reg = _transpose_and_gather_feat(t_reg, t_inds)
            t_reg = t_reg.view(batch, 1, 1, K3, 1, 1, 2)
            fo_reg = _transpose_and_gather_feat(fo_reg, fo_inds)
            fo_reg = fo_reg.view(batch, 1, 1, 1, K4, 1, 2)
            fi_reg = _transpose_and_gather_feat(fi_reg, fi_inds)
            fi_reg = fi_reg.view(batch, 1, 1, 1, 1, K5, 2)

            f_xs = f_xs + f_reg[..., 0]
            f_ys = f_ys + f_reg[..., 1]
            s_xs = s_xs + s_reg[..., 0]
            s_ys = s_ys + s_reg[..., 1]
            t_xs = t_xs + t_reg[..., 0]
            t_ys = t_ys + t_reg[..., 1]
            fo_xs = fo_xs + fo_reg[..., 0]
            fo_ys = fo_ys + fo_reg[..., 1]
            fi_xs = fi_xs + fi_reg[..., 0]
            fi_ys = fi_ys + fi_reg[..., 1]

        kps = torch.stack((f_xs, f_ys, s_xs, s_ys, t_xs, t_ys, fo_xs, fo_ys, fi_xs, fi_ys), dim=6)
        kps = kps.view(batch, -1, 10)
        kps = _gather_feat(kps, inds)

        tag_1 = f_tag.contiguous().view(batch, -1, 1)
        tag_1 = _gather_feat(tag_1, inds).float()
        tag_2 = s_tag.contiguous().view(batch, -1, 1)
        tag_2 = _gather_feat(tag_2, inds).float()
        tag_3 = t_tag.contiguous().view(batch, -1, 1)
        tag_3 = _gather_feat(tag_3, inds).float()
        tag_4 = fo_tag.contiguous().view(batch, -1, 1)
        tag_4 = _gather_feat(tag_4, inds).float()
        tag_5 = fi_tag.contiguous().view(batch, -1, 1)
        tag_5 = _gather_feat(tag_5, inds).float()

        clses_1 = f_clses.contiguous().view(batch, -1, 1)
        clses_1 = _gather_feat(clses_1, inds).float()
        clses_2 = s_clses.contiguous().view(batch, -1, 1)
        clses_2 = _gather_feat(clses_2, inds).float()
        clses_3 = t_clses.contiguous().view(batch, -1, 1)
        clses_3 = _gather_feat(clses_3, inds).float()
        clses_4 = fo_clses.contiguous().view(batch, -1, 1)
        clses_4 = _gather_feat(clses_4, inds).float()
        clses_5 = fi_clses.contiguous().view(batch, -1, 1)
        clses_5 = _gather_feat(clses_5, inds).float()

        det_kps = torch.cat([kps, scores, clses_1, clses_2, clses_3, clses_4, clses_5, tag_1, tag_2, tag_3, tag_4, tag_5], dim=2)

    det_kps = det_kps.detach().cpu().numpy()
    det_kps = det_kps.reshape(1, -1, det_kps.shape[2])

    c = np.array([input_w / 2., input_h / 2.], dtype=np.float32)
    s = max(input_h, input_w) * 1.0

    out_height = input_h // 4
    out_width = input_w // 4

    c = [c]
    s = [s]
    for i in range(det_kps.shape[0]):
        det_kps[i, :, 0:2] = transform_preds(
            det_kps[i, :, 0:2], c[i], s[i], (out_width, out_height))
        det_kps[i, :, 2:4] = transform_preds(
            det_kps[i, :, 2:4], c[i], s[i], (out_width, out_height))
        det_kps[i, :, 4:6] = transform_preds(
            det_kps[i, :, 4:6], c[i], s[i], (out_width, out_height))
        det_kps[i, :, 6:8] = transform_preds(
            det_kps[i, :, 6:8], c[i], s[i], (out_width, out_height))
        det_kps[i, :, 8:10] = transform_preds(
            det_kps[i, :, 8:10], c[i], s[i], (out_width, out_height))

    return det_kps

def filter_overlap_hl(scores, inds, clses, ys, xs):
    # turn tensor to array
    scores = scores.cpu().numpy()
    inds = inds.cpu().numpy()
    clses = clses.cpu().numpy()
    ys = ys.cpu().numpy()
    xs = xs.cpu().numpy()

    MAX_AFF = 3

    out_scores = []
    out_inds = []
    out_clses = []
    out_ys = []
    out_xs = []
    for idx in range(scores.shape[1]):
        flag = True
        for idx1 in range(len(out_ys)):
            if np.sqrt((out_xs[idx1] - xs[0, idx])**2 + (out_ys[idx1] - ys[0, idx])**2) < np.sqrt(2.1):
                flag = False
        if flag:
            if np.where(out_clses == clses[0, idx])[0].size < MAX_AFF:
                out_scores.append(scores[0, idx])
                out_inds.append(inds[0, idx])
                out_clses.append(clses[0, idx])
                out_ys.append(ys[0, idx])
                out_xs.append(xs[0, idx])

    K = len(out_scores)
    out_scores = np.expand_dims(np.array(out_scores, dtype=scores.dtype), axis=0)
    out_inds = np.expand_dims(np.array(out_inds, dtype=inds.dtype), axis=0)
    out_clses = np.expand_dims(np.array(out_clses, dtype=clses.dtype), axis=0)
    out_ys = np.expand_dims(np.array(out_ys, dtype=ys.dtype), axis=0)
    out_xs = np.expand_dims(np.array(out_xs, dtype=xs.dtype), axis=0)


    return torch.from_numpy(out_scores).cuda(), torch.from_numpy(out_inds).long().cuda(), \
           torch.from_numpy(out_clses).cuda(), torch.from_numpy(out_ys).cuda(), \
           torch.from_numpy(out_xs).cuda(), K

def single_aff_filter(kps):
    out_kps = []
    aff_list = np.unique(kps[:, 11])
    for aff in aff_list:
        idx_1 = np.where(kps[:, 11] == aff)
        out_kps.append(kps[idx_1[0][0], :])

    out_dict = {}
    for kps in out_kps:
        out_dict[kps[11]] = kps[:10]

    return out_dict


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    eval_multi_models(args)
