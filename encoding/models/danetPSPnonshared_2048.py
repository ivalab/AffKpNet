###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize
from ..nn import PAM_Module
from ..nn import CAM_Module
from ..models import BaseNet


__all__ = ['DANetPSP', 'get_danetpsp']

class DANetPSP(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANetPSP, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return tuple(outputs)
        
class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa1x1 = PAM_Module(inter_channels)
        self.sa2x2 = PAM_Module(inter_channels)
        self.sa3x3 = PAM_Module(inter_channels)
        self.sa6x6 = PAM_Module(inter_channels)


        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.avepool1x1 = nn.AvgPool2d(20, 20)
        #self.conv1x1 = nn.Sequential(nn.Conv2d(512, 128, 1), norm_layer(inter_channels//4),nn.Dropout2d(0.1, False), )

        self.avepool2x2 = nn.AvgPool2d(10, 10)
        #self.conv2x2 = nn.Sequential(nn.Conv2d(512, 128, 1), norm_layer(inter_channels//4),nn.Dropout2d(0.1, False), )

        self.avepool3x3 = nn.AvgPool2d(5, 5)
        #self.conv3x3 = nn.Sequential(nn.Conv2d(512, 128, 1), norm_layer(inter_channels//4),nn.Dropout2d(0.1, False), )

        self.avepool6x6 = nn.AvgPool2d(2, 2)
        #self.conv6x6 = nn.Sequential(nn.Conv2d(512, 128, 1), norm_layer(inter_channels//4),nn.Dropout2d(0.1, False), )

        self.interp = nn.Upsample(size=(60, 60), mode='bilinear')
        self.conv512x512 = nn.Sequential(nn.Conv2d(512, 2048, 1), norm_layer(inter_channels*4),nn.Dropout2d(0.1, False), )
        self.conv512 = nn.Sequential(nn.Conv2d(4096, 512, 1), norm_layer(inter_channels),nn.Dropout2d(0.1, False), )


    def forward(self, x):
        feat1 = self.conv5a(x)

        # PSP below
        featpsp1x1 = self.avepool1x1(feat1)
        featpsp1x1 = self.sa1x1(featpsp1x1)
        sa_feat1x1 = self.interp(featpsp1x1)
        #sa_feat1x1 = self.conv1x1(featpsp1x1)

        featpsp2x2 = self.avepool2x2(feat1)
        featpsp2x2 = self.sa2x2(featpsp2x2)
        sa_feat2x2 = self.interp(featpsp2x2)
        #sa_feat2x2 = self.conv2x2(featpsp2x2)

        featpsp3x3 = self.avepool3x3(feat1)
        featpsp3x3 = self.sa3x3(featpsp3x3)
        sa_feat3x3 = self.interp(featpsp3x3)
        #sa_feat3x3 = self.conv3x3(featpsp3x3)

        featpsp6x6 = self.avepool6x6(feat1)
        featpsp6x6 = self.sa6x6(featpsp6x6)
        sa_feat6x6 = self.interp(featpsp6x6)
        #sa_feat6x6 = self.conv6x6(featpsp6x6)

        sa_feat = self.sa(feat1)

        # concatenate all 4 channels
        sa_feat = self.conv512x512(sa_feat)
        sa_feat = torch.cat((sa_feat,sa_feat1x1,sa_feat2x2,sa_feat3x3,sa_feat6x6), 1)
        sa_feat = self.conv512(sa_feat)

        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)


def get_danetpsp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='./pretrain_models', **kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ..datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = DANetPSP(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

