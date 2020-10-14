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

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses, BatchNorm2d
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset, test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from option import Options
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

import scipy.io as sio
# SAVE_ROOT = '/media/fujenchu/home3/data/affordanceSeg/SRFAff_v1/data/Affordance_Part_Data'
SAVE_ROOT = '/home/fujenchu/projects/affordanceContext/DANet/datasets/UMD_affordance'
SAVE_FOLDER_NAME = 'danetpspsharedbn_bg'
SAVE_FOLDER_NAME = 'danet_umd_ours'
SAVE_FOLDER_NAME = 'pred_seg_kp_umdself'

color_dict = {1: [0,0,205], #grasp red
              2: [34,139,34], #cut green
              3: [0,255,255], #scoop bluegreen
              4: [165,42,42], #contain dark blue
              5: [128,64,128], #pound purple
              6: [184,134,11]} #wrap-grasp light blue

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
    def eval_batch(image, dst, img_path, model, eval_mode):
        if eval_mode:
            # evaluation mode on validation set
            targets = dst
            outputs = model(image)

            # create save path
            savename = img_path[0].split('/')[-1]
            savename_splits = savename.split('_')
            save_obj = savename_splits[0]
            save_objsub = savename_splits[0] + '_' + savename_splits[1]
            save_mat = savename_splits[0] + '_' + savename_splits[1] + '_' + savename_splits[2] + '_rgbd_pred.mat'

            SAVE_FOLDER = os.path.join(SAVE_ROOT, SAVE_FOLDER_NAME, save_obj, save_objsub)
            if not os.path.exists(SAVE_FOLDER):
                os.makedirs(SAVE_FOLDER)

            # save mat file
            output_np = np.asarray(outputs['sasc'].data.cpu())
            output_np = output_np[0].transpose(1, 2, 0)
            output_np = np.asarray(np.argmax(output_np, axis=2), dtype=np.uint8)

            sio.savemat(os.path.join(SAVE_FOLDER, save_mat), {'pred_label': output_np}, do_compression=True)

    for i, (image, dst, img_path) in enumerate(tbar):
        with torch.no_grad():
            image = image.cuda()
            for key, value in dst.items():
                if not dst[key].is_cuda:
                    dst[key] = value.cuda()
            eval_batch(image, dst, img_path, model, args.eval)

def eval_models(args):
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

    print('Saving affordance mask prediction is finished!!!')

if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    eval_models(args)
