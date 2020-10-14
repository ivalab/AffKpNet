###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data

__all__ = ['BaseDataset', 'test_batchify_fn']

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None, 
                 target_transform=None, base_size=520, crop_size=480,
                 logger=None, scale=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.logger = logger
        self.scale = scale

        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

        if not self.scale:
            if self.logger is not None:
                self.logger.info('single scale training!!!')


    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        if self.scale:
            #short_size = random.randint(int(self.base_size*0.75), int(self.base_size*2.0))
            short_size = random.randint(int(self.base_size * 0.80), int(self.base_size * 1.20))
        else:
            short_size = self.base_size
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        # deg = random.uniform(-10, 10)
        # img = img.rotate(deg, resample=Image.BILINEAR)
        # mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)#pad 0 for umd
        # pad image at right and bottom region won't affect the keypoint location

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        # if random.random() < 0.5:
        #     img = img.filter(ImageFilter.GaussianBlur(
        #         radius=random.random()))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


class BaseKpDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None,
                 target_transform=None, base_size=520, crop_size=480,
                 logger=None, scale=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.logger = logger
        self.scale = scale

        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

        if not self.scale:
            if self.logger is not None:
                self.logger.info('single scale training!!!')


    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def _val_sync_transform(self, img, mask, kps):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        for kp in kps:
            kp[1] = [int(ow / w * kp[1][0]), int(oh / h * kp[1][1])]
            kp[2] = [int(ow / w * kp[2][0]), int(oh / h * kp[2][1])]
            kp[3] = [int(ow / w * kp[3][0]), int(oh / h * kp[3][1])]
            kp[4] = [int(ow / w * kp[4][0]), int(oh / h * kp[4][1])]
            kp[5] = [int(ow / w * kp[5][0]), int(oh / h * kp[5][1])]

        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))

        for kp in kps:
            kp[1] = [int(kp[1][0] - x1), int(kp[1][1] - y1)]
            kp[2] = [int(kp[2][0] - x1), int(kp[2][1] - y1)]
            kp[3] = [int(kp[3][0] - x1), int(kp[3][1] - y1)]
            kp[4] = [int(kp[4][0] - x1), int(kp[4][1] - y1)]
            kp[5] = [int(kp[5][0] - x1), int(kp[5][1] - y1)]

        for kp in kps:
            if not kp[1][0] in range(0, self.crop_size) or not kp[1][1] in range(0, self.crop_size) or \
                not kp[2][0] in range(0, self.crop_size) or not kp[2][1] in range(0, self.crop_size) or \
                not kp[3][0] in range(0, self.crop_size) or not kp[3][1] in range(0, self.crop_size) or \
                not kp[4][0] in range(0, self.crop_size) or not kp[4][1] in range(0, self.crop_size) or \
                not kp[5][0] in range(0, self.crop_size) or not kp[5][1] in range(0, self.crop_size):

                print('keypoint is outside of the bound')

        # final transform
        return img, self._mask_transform(mask), kps

    def _sync_transform(self, img, mask, kps):
        flag_kp_w_bound = False

        while not flag_kp_w_bound:
            out_kps = kps.copy()
            # turn the flag on, turn it off if any keypoint fall outside of bound after transformation
            flag_kp_w_bound = True

            w, h = img.size

            # random mirror
            flag_mirror = random.random()
            if flag_mirror < 0.5:
                # flip the kp
                for kp in out_kps:
                    kp[1] = [int(w - kp[1][0]), int(kp[1][1])]
                    kp[2] = [int(w - kp[2][0]), int(kp[2][1])]
                    kp[3] = [int(w - kp[3][0]), int(kp[3][1])]
                    kp[4] = [int(w - kp[4][0]), int(kp[4][1])]
                    kp[5] = [int(w - kp[5][0]), int(kp[5][1])]

                    # debug

            # determine resize related parameters
            if self.scale:
                #short_size = random.randint(int(self.base_size*0.75), int(self.base_size*2.0))
                short_size = random.randint(int(self.base_size * 0.80), int(self.base_size * 1.20))
            else:
                short_size = self.base_size

            if h > w:
                ow = short_size
                oh = int(1.0 * h * ow / w)
            else:
                oh = short_size
                ow = int(1.0 * w * oh / h)

            for kp in out_kps:
                kp[1] = [int(ow / w * kp[1][0]), int(oh / h * kp[1][1])]
                kp[2] = [int(ow / w * kp[2][0]), int(oh / h * kp[2][1])]
                kp[3] = [int(ow / w * kp[3][0]), int(oh / h * kp[3][1])]
                kp[4] = [int(ow / w * kp[4][0]), int(oh / h * kp[4][1])]
                kp[5] = [int(ow / w * kp[5][0]), int(oh / h * kp[5][1])]

            # debug


            # random rotate -10~10, mask using NN rotate
            # deg = random.uniform(-10, 10)
            # img = img.rotate(deg, resample=Image.BILINEAR)
            # mask = mask.rotate(deg, resample=Image.NEAREST)
            # pad crop

            # determine crop related parameters
            crop_size = self.crop_size

            # random crop crop_size
            w, h = img.size
            x1 = random.randint(0, w - crop_size)
            y1 = random.randint(0, h - crop_size)

            for kp in out_kps:
                kp[1] = [int(kp[1][0] - x1), int(kp[1][1]) - y1]
                kp[2] = [int(kp[2][0] - x1), int(kp[2][1]) - y1]
                kp[3] = [int(kp[3][0] - x1), int(kp[3][1]) - y1]
                kp[4] = [int(kp[4][0] - x1), int(kp[4][1]) - y1]
                kp[5] = [int(kp[5][0] - x1), int(kp[5][1]) - y1]

            for kp in out_kps:
                if not kp[1][0] in range(0, crop_size) or not kp[1][1] in range(0, crop_size) or \
                    not kp[2][0] in range(0, crop_size) or not kp[2][1] in range(0, crop_size) or \
                    not kp[3][0] in range(0, crop_size) or not kp[3][1] in range(0, crop_size) or \
                    not kp[4][0] in range(0, crop_size) or not kp[4][1] in range(0, crop_size) or \
                    not kp[5][0] in range(0, crop_size) or not kp[5][1] in range(0, crop_size):
                    flag_kp_w_bound = False
                    break

            # gaussian blur as in PSP
            # if random.random() < 0.5:
            #     img = img.filter(ImageFilter.GaussianBlur(
            #         radius=random.random()))

        # random mirror the image and mask
        if flag_mirror < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # resize image and mask
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # pad image if image is smaller than crop size after resize
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)  # pad 0 for umd
            # pad image at right and bottom region won't affect the keypoint location

        # crop image and mask
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

        # final transform
        return img, self._mask_transform(mask), out_kps

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()

def test_batchify_fn(data):
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], (tuple, list)):
        data = zip(*data)
        return [test_batchify_fn(i) for i in data]
    elif isinstance(data[0], dict):
        return list(data)
    raise TypeError((error_msg.format(type(batch[0]))))

