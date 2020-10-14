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


__all__ = ['DANetKp', 'get_danet_kp']

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class DANetKp(BaseNet):
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
        super(DANetKp, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = DANetHead(2048, nclass, norm_layer)

        self.inplanes = 2048
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            1,
            [256],
            [4],
        )
        # self.deconv_layers = self._make_deconv_layer(
        #     3,
        #     [256, 256, 256],
        #     [4, 4, 4],
        # )
        self.head_conv = 64


        self.heads = {'1': nclass-1, '1_reg': 2, '1_tag': 1,
                      '2': nclass-1, '2_reg': 2, '2_tag': 1,
                      '3': nclass-1, '3_reg': 2, '3_tag': 1,
                      '4': nclass-1, '4_reg': 2, '4_tag': 1,
                      '5': nclass-1, '5_reg': 2, '5_tag': 1,}
        # self.heads = {'1': nclass, '1_reg': 2, '1_tag': 1,
        #               '2': nclass, '2_reg': 2, '2_tag': 1,
        #               '3': nclass, '3_reg': 2, '3_tag': 1,
        #               '4': nclass, '4_reg': 2, '4_tag': 1,
        #               '5': nclass, '5_reg': 2, '5_tag': 1, }

        # for keypoint detection
        for head in sorted(self.heads):
            num_output = self.heads[head]
            if self.head_conv > 0:
                # if 'tag' in head or 'reg' in head:
                #     fc = nn.Sequential(
                #         nn.Conv2d(256, 128,
                #                   kernel_size=3, stride=1),
                #         nn.BatchNorm2d(128, momentum=0.1),
                #         nn.Conv2d(128, 64,
                #                   kernel_size=3, stride=1),
                #         nn.BatchNorm2d(64, momentum=0.1),
                #         nn.Conv2d(64, self.head_conv,
                #                   kernel_size=3, padding=1, bias=True),
                #         nn.ReLU(inplace=True),
                #         nn.Conv2d(self.head_conv, num_output,
                #                   kernel_size=1, stride=1, padding=0))
                # else:
                fc = nn.Sequential(
                    nn.Conv2d(256, self.head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.head_conv, num_output,
                              kernel_size=1, stride=1, padding=0))
                if '1' in head or '2' in head or '3' in head or '4' in head or '5' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            self.__setattr__(head, fc)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}

        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        # keypoint detection
        y = self.deconv_layers(c4)
        for head in self.heads:
            outputs[head] = self.__getattr__(head)(y)

        # segmentation
        x = self.head(c4)
        x = list(x)

        x[0] = upsample(x[0], imsize, **self._up_kwargs) #ToDo, confirm the downsample factor
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        # outputs = [x[0]]
        # outputs.append(x[1])
        # outputs.append(x[2])
        outputs['sasc'] = x[0]
        outputs['sa'] = x[1]
        outputs['sc'] = x[2]

        return outputs
        
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

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
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


def get_danet_kp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
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
    model = DANetKp(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

