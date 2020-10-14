##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""Encoding Custermized NN Module"""
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

torch_ver = torch.__version__[:3]

__all__ = ['GramMatrix', 'SegmentationLosses', 'View', 'Sum', 'Mean',
           'Normalize', 'PyramidPooling','SegmentationMultiLosses', 'SegMultiKpLosses']

class GramMatrix(Module):
    r""" Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \mathcal{G} = \sum_{h=1}^{H_i}\sum_{w=1}^{W_i} \mathcal{F}_{h,w}\mathcal{F}_{h,w}^T
    """
    def forward(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

def softmax_crossentropy(input, target, weight, size_average, ignore_index, reduce=True):
    return F.nll_loss(F.log_softmax(input, 1), target, weight,
                      size_average, ignore_index, reduce)

class SegmentationLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.2, weight=None,
                 size_average=True, ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, size_average, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = BCELoss(weight, size_average) 

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            return super(SegmentationLosses, self).forward(*inputs)
        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2
        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2
        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(F.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(), 
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect

def _ae_loss_5(tag1, tag2, tag3, tag4, tag5, mask):
    num  = mask.sum(dim=1, keepdim=True).float()
    tag1 = tag1.squeeze()
    tag2 = tag2.squeeze()
    tag3 = tag3.squeeze()
    tag4 = tag4.squeeze()
    tag5 = tag5.squeeze()

    tag_mean = (tag1 + tag2 + tag3 + tag4 + tag5) / 5

    tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
    # print("tag:{} mask:{}".format(tag1, mask))
    tag1 = tag1[mask].sum()
    tag2 = torch.pow(tag2 - tag_mean, 2) / (num + 1e-4)
    tag2 = tag2[mask].sum()
    tag3 = torch.pow(tag3 - tag_mean, 2) / (num + 1e-4)
    tag3 = tag3[mask].sum()
    tag4 = torch.pow(tag4 - tag_mean, 2) / (num + 1e-4)
    tag4 = tag4[mask].sum()
    tag5 = torch.pow(tag5 - tag_mean, 2) / (num + 1e-4)
    tag5 = tag5[mask].sum()
    pull = tag1 + tag2 + tag3 + tag4 + tag5

    mask = mask.unsqueeze(1) + mask.unsqueeze(2)
    mask = mask.eq(2)
    num  = num.unsqueeze(2)
    num2 = (num - 1) * num
    if len(tag_mean.size()) < 2:
      tag_mean = tag_mean.unsqueeze(0)
    dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
    dist = 1 - torch.abs(dist)
    dist = nn.functional.relu(dist, inplace=True)
    dist = dist - 1 / (num + 1e-4)
    dist = dist / (num2 + 1e-4)
    dist = dist[mask]
    push = dist.sum()
    return pull, push

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class AeLoss(nn.Module):
  '''nn.Module warpper for associate embedding loss'''
  def __init__(self):
      super(AeLoss, self).__init__()
      self.ae_loss = _ae_loss_5

  def forward(self, tag1, tag2, tag3, tag4, tag5, ind1, ind2, ind3, ind4, ind5, mask):
      tag1 = _transpose_and_gather_feat(tag1, ind1)
      tag2 = _transpose_and_gather_feat(tag2, ind2)
      tag3 = _transpose_and_gather_feat(tag3, ind3)
      tag4 = _transpose_and_gather_feat(tag4, ind4)
      tag5 = _transpose_and_gather_feat(tag5, ind5)

      return self.ae_loss(tag1, tag2, tag3, tag4, tag5, mask)

def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  # pos_weight = neg_inds.float().sum()
  # neg_weight = pos_inds.float().sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    # loss = loss - (pos_loss * pos_weight + neg_loss * neg_weight) / num_pos
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

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

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = loss / (mask.sum() + 1e-4)
        return loss

class SegMultiKpLosses(nn.Module):
    def __init__(self, pull_weight=1.0, push_weight=1.0, regr_weight=1.0, seg_weight=1.0):
        super(SegMultiKpLosses, self).__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.regr_weight = regr_weight
        self.seg_weight = seg_weight

        self.crit = FocalLoss()
        # self.crit = torch.nn.MSELoss()
        self.crit_reg = RegL1Loss()
        self.crit_tag = AeLoss()
        self.crit_seg = CrossEntropyLoss()

    def forward(self, outputs, batch):
        outputs['1'] = _sigmoid(outputs['1'])
        outputs['2'] = _sigmoid(outputs['2'])
        outputs['3'] = _sigmoid(outputs['3'])
        outputs['4'] = _sigmoid(outputs['4'])
        outputs['5'] = _sigmoid(outputs['5'])

        # segmentation loss
        loss_seg_sasc = self.crit_seg(outputs['sasc'], batch['seg'])
        loss_seg_sa = self.crit_seg(outputs['sa'], batch['seg'])
        loss_seg_sc = self.crit_seg(outputs['sc'], batch['seg'])
        seg_loss = loss_seg_sasc + loss_seg_sa + loss_seg_sc

        # focal loss
        f_focal_loss = self.crit(outputs['1'], batch['1'])
        s_focal_loss = self.crit(outputs['2'], batch['2'])
        t_focal_loss = self.crit(outputs['3'], batch['3'])
        fo_focal_loss = self.crit(outputs['4'], batch['4'])
        fi_focal_loss = self.crit(outputs['5'], batch['5'])
        focal_loss = f_focal_loss + s_focal_loss + t_focal_loss + fo_focal_loss + fi_focal_loss

        # tag loss
        pull, push = self.crit_tag(outputs['1_tag'], outputs['2_tag'], outputs['3_tag'], outputs['4_tag'], outputs['5_tag'],
                                   batch['1_tag'], batch['2_tag'], batch['3_tag'], batch['4_tag'], batch['5_tag'],
                                   batch['reg_mask'])
        pull_loss = self.pull_weight * pull
        push_loss = self.push_weight * push

        # reg loss
        f_reg_loss = self.regr_weight * self.crit_reg(outputs['1_reg'], batch['reg_mask'], batch['1_tag'],
                                                     batch['1_reg'])
        s_reg_loss = self.regr_weight * self.crit_reg(outputs['2_reg'], batch['reg_mask'], batch['2_tag'],
                                                     batch['2_reg'])
        t_reg_loss = self.regr_weight * self.crit_reg(outputs['3_reg'], batch['reg_mask'], batch['3_tag'],
                                                     batch['3_reg'])
        fo_reg_loss = self.regr_weight * self.crit_reg(outputs['4_reg'], batch['reg_mask'], batch['4_tag'],
                                                      batch['4_reg'])
        fi_reg_loss = self.regr_weight * self.crit_reg(outputs['5_reg'], batch['reg_mask'], batch['5_tag'],
                                                      batch['5_reg'])
        reg_loss = f_reg_loss + s_reg_loss + t_reg_loss + fo_reg_loss + fi_reg_loss

        loss = self.seg_weight * seg_loss + focal_loss + reg_loss + pull_loss + push_loss
        # loss = focal_loss + reg_loss + pull_loss + push_loss
        # loss = seg_loss

        loss_stat = {}
        loss_stat['seg_loss'] = seg_loss
        loss_stat['focal_loss'] = focal_loss
        loss_stat['reg_loss'] = reg_loss
        loss_stat['pull_loss'] = pull_loss
        loss_stat['push_loss'] = push_loss

        return loss, loss_stat

class SegmentationMultiLosses(CrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-L1oss"""
    def __init__(self, nclass=-1, weight=None,size_average=True, ignore_index=-1):
        super(SegmentationMultiLosses, self).__init__(weight, size_average, ignore_index)
        self.nclass = nclass


    def forward(self, *inputs):
        *preds, target = tuple(inputs)
        pred1, pred2 ,pred3= tuple(preds[0])


        loss1 = super(SegmentationMultiLosses, self).forward(pred1, target)
        loss2 = super(SegmentationMultiLosses, self).forward(pred2, target)
        loss3 = super(SegmentationMultiLosses, self).forward(pred3, target)
        loss = loss1 + loss2 + loss3
        return loss


class View(Module):
    """Reshape the input into different size, an inplace operator, support
    SelfParallel mode.
    """
    def __init__(self, *args):
        super(View, self).__init__()
        if len(args) == 1 and isinstance(args[0], torch.Size):
            self.size = args[0]
        else:
            self.size = torch.Size(args)

    def forward(self, input):
        return input.view(self.size)


class Sum(Module):
    def __init__(self, dim, keep_dim=False):
        super(Sum, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.sum(self.dim, self.keep_dim)


class Mean(Module):
    def __init__(self, dim, keep_dim=False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keep_dim = keep_dim

    def forward(self, input):
        return input.mean(self.dim, self.keep_dim)


class Normalize(Module):
    r"""Performs :math:`L_p` normalization of inputs over specified dimension.

    Does:

    .. math::
        v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}

    for each subtensor v over dimension dim of input. Each subtensor is
    flattened into a vector, i.e. :math:`\lVert v \rVert_p` is not a matrix
    norm.

    With default arguments normalizes over the second dimension with Euclidean
    norm.

    Args:
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
    """
    def __init__(self, p=2, dim=1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, self.p, self.dim, eps=1e-8)


class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)



