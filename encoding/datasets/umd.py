###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################

import os
import sys
import numpy as np
import random
import math
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageDraw

import torch
import torch.utils.data as data
import torchvision.transforms as transform
import re
from tqdm import tqdm
from .base import BaseDataset, BaseKpDataset

class UMDSelfKpSegmentation(BaseKpDataset):
    BASE_DIR = 'UMD_affordance'
    NUM_CLASS = 7
    DownSample_Factor = 4

    def __init__(self, root='../datasets', split='train',  # TODO what is split here
                 mode=None, transform=None, target_transform=None, **kwargs):
        super(UMDSelfKpSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)
        # assert exists
        root = os.path.join(root, self.BASE_DIR)
        assert os.path.exists(root), "Please download the dataset!!"

        self.max_objs = 3
        self.dict = {'grasp': 1,
                     'cut': 2,
                     'scoop': 3,
                     'contain': 4,
                     'pound': 5,
                     'wgrasp': 6}

        self.images, self.masks, self.kps = _get_umd_self_triplets(root, split, mode)
        if split != 'vis':
            assert (len(self.images) == len(self.masks)) and (len(self.images) == len(self.kps)) and (len(self.kps) == len(self.masks))
        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))


    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'vis':
            if self.transform is not None:
                img = self.transform(img)
            print(os.path.basename(self.images[index]))
            return img, os.path.basename(self.images[index])

        mask = Image.open(self.masks[index])

        # load heatmap for keypoint
        anns_kp = []
        file_kp = open(self.kps[index], 'r')
        affordance = file_kp.readline()
        while affordance:
            line1 = file_kp.readline().split()
            p1 = [int(float(line1[0])), int(float(line1[1]))]
            line2 = file_kp.readline().split()
            p2 = [int(float(line2[0])), int(float(line2[1]))]
            line3 = file_kp.readline().split()
            p3 = [int(float(line3[0])), int(float(line3[1]))]
            line4 = file_kp.readline().split()
            p4 = [int(float(line4[0])), int(float(line4[1]))]
            line5 = file_kp.readline().split()
            p5 = [int(float(line5[0])), int(float(line5[1]))]
            anns_kp.append([self.dict[affordance.split('\n')[0]], p1, p2, p3, p4, p5])

            affordance = file_kp.readline()
        file_kp.close()

        # synchrosized transform
        if self.mode == 'train':
            img, mask, kps = self._sync_transform(img, mask, anns_kp)
        elif self.mode == 'val':
            img, mask, kps = self._val_sync_transform(img, mask, anns_kp)
        elif self.mode == 'testval_kp':
            mask = self._mask_transform(mask)
            kps = anns_kp
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
            kps = anns_kp


        output_w = int(self.crop_size / self.DownSample_Factor)
        output_h = int(self.crop_size / self.DownSample_Factor)
        # create target for keypoints
        f_heatmaps = np.zeros((self.NUM_CLASS - 1, output_h, output_w), dtype=np.float32)
        s_heatmaps = np.zeros((self.NUM_CLASS - 1, output_h, output_w), dtype=np.float32)
        t_heatmaps = np.zeros((self.NUM_CLASS - 1, output_h, output_w), dtype=np.float32)
        fo_heatmaps = np.zeros((self.NUM_CLASS - 1, output_h, output_w), dtype=np.float32)
        fi_heatmaps = np.zeros((self.NUM_CLASS - 1, output_h, output_w), dtype=np.float32)


        f_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        s_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        t_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        fo_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        fi_reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        f_tag = np.zeros((self.max_objs), dtype=np.int64)
        s_tag = np.zeros((self.max_objs), dtype=np.int64)
        t_tag = np.zeros((self.max_objs), dtype=np.int64)
        fo_tag = np.zeros((self.max_objs), dtype=np.int64)
        fi_tag = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        draw_gaussian = draw_umich_gaussian
        for k, ann_kp in enumerate(kps):
            # bg class isn't counted in kp detection
            cls_id = ann_kp[0] - 1

            f_p1 = np.array([ann_kp[1][0] / float(self.DownSample_Factor), ann_kp[1][1] / float(self.DownSample_Factor)])
            f_p2 = np.array([ann_kp[2][0] / float(self.DownSample_Factor), ann_kp[2][1] / float(self.DownSample_Factor)])
            f_p3 = np.array([ann_kp[3][0] / float(self.DownSample_Factor), ann_kp[3][1] / float(self.DownSample_Factor)])
            f_p4 = np.array([ann_kp[4][0] / float(self.DownSample_Factor), ann_kp[4][1] / float(self.DownSample_Factor)])
            f_p5 = np.array([ann_kp[5][0] / float(self.DownSample_Factor), ann_kp[5][1] / float(self.DownSample_Factor)])

            p1 = f_p1.astype(np.int32)
            p2 = f_p2.astype(np.int32)
            p3 = f_p3.astype(np.int32)
            p4 = f_p4.astype(np.int32)
            p5 = f_p5.astype(np.int32)

            # no.1 point
            dst_1 = np.sqrt(np.sum(np.power(p1 - p5, 2)))
            radius1 = gaussian_radius((math.ceil(dst_1), math.ceil(dst_1)))
            radius1 = max(0, int(radius1))
            radius1 /= 2

            draw_gaussian(f_heatmaps[cls_id], p1, int(radius1))

            # no.2 point
            dst_2 = np.sqrt(np.sum(np.power(p2 - p5, 2)))
            radius2 = gaussian_radius((math.ceil(dst_2), math.ceil(dst_2)))
            radius2 = max(0, int(radius2))
            radius2 /= 2

            draw_gaussian(s_heatmaps[cls_id], p2, int(radius2))

            # no.3 point
            dst_3 = np.sqrt(np.sum(np.power(p3 - p5, 2)))
            radius3 = gaussian_radius((math.ceil(dst_3), math.ceil(dst_3)))
            radius3 = max(0, int(radius3))
            radius3 /= 2

            draw_gaussian(t_heatmaps[cls_id], p3, int(radius3))

            # no.4 point
            dst_4 = np.sqrt(np.sum(np.power(p4 - p5, 2)))
            radius4 = gaussian_radius((math.ceil(dst_4), math.ceil(dst_4)))
            radius4 = max(0, int(radius4))
            radius4 /= 2

            draw_gaussian(fo_heatmaps[cls_id], p4, int(radius4))

            # no.5 point
            dst_5 = np.min([dst_1, dst_2, dst_3, dst_4])
            radius5 = gaussian_radius((math.ceil(dst_5), math.ceil(dst_5)))
            radius5 = max(0, int(radius5))
            radius5 /= 2

            draw_gaussian(fi_heatmaps[cls_id], p5, int(radius5))

            f_tag[k] = p1[1] * output_w + p1[0]
            s_tag[k] = p2[1] * output_w + p2[0]
            t_tag[k] = p3[1] * output_w + p3[0]
            fo_tag[k] = p4[1] * output_w + p4[0]
            fi_tag[k] = p5[1] * output_w + p5[0]

            f_reg[k] = f_p1 - p1
            s_reg[k] = f_p2 - p2
            t_reg[k] = f_p3 - p3
            fo_reg[k] = f_p4 - p4
            fi_reg[k] = f_p5 - p5

            reg_mask[k] = 1


        # general resize, normalize and toTensor
        if self.transform is not None:
            img_trans = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)

        target = {
            'seg': mask,
            '1': f_heatmaps, '2': s_heatmaps, '3': t_heatmaps, '4': fo_heatmaps, '5': fi_heatmaps,
            '1_tag': f_tag, '2_tag': s_tag, '3_tag': t_tag, '4_tag': fo_tag, '5_tag': fi_tag,
            '1_reg': f_reg, '2_reg': s_reg, '3_reg': t_reg, '4_reg': fo_reg, '5_reg': fi_reg,
            'reg_mask': reg_mask}

        if self.mode == 'testval':
            return img_trans, target, self.masks[index]  # only for segmentation evaluation
        elif self.mode == 'testval_kp':
            return np.array(img), img_trans, target, self.masks[index]  # only for evaluation
        else:
            return img_trans, target


    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

def _get_umd_self_triplets(folder, split='train', mode='train'):
    def get_path_triplets(folder, split_f):
        img_paths = []
        mask_paths = []
        kp_paths = []
        with open(split_f, 'r') as lines:
            for line in tqdm(lines):
                ll_str = re.split('\t', line)
                imgpath = os.path.join(folder, ll_str[0].rstrip())
                maskpath = os.path.join(folder, ll_str[1].rstrip())
                kppath = os.path.join(folder, ll_str[2].rstrip())
                if os.path.isfile(maskpath):
                    img_paths.append(imgpath)
                    mask_paths.append(maskpath)
                    kp_paths.append(kppath)
                else:
                    print('cannot find the mask:', maskpath)

        return img_paths[:], mask_paths[:], kp_paths[:]

    if split == 'train':
        split_f = os.path.join(folder, 'train_umdself_kp_cateSplit.txt')
        img_paths, mask_paths, kp_paths = get_path_triplets(folder, split_f)
    elif split == 'val':
        split_f = os.path.join(folder, 'val_umdself_kp_cateSplit.txt')
        img_paths, mask_paths, kp_paths = get_path_triplets(folder, split_f)
    elif split == 'test':
        split_f = os.path.join(folder, 'test_umdself_kp_cateSplit.txt')
        img_paths, mask_paths, kp_paths = get_path_triplets(folder, split_f)
    else:
        split_f = os.path.join(folder, 'trainval_fine.txt')
        img_paths, mask_paths, kp_paths = get_path_triplets(folder, split_f)

    return img_paths, mask_paths, kp_paths

def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
