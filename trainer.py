#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
import os
import numpy as np
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from torch import nn
import torch as t
from utils import array_tool as at
from utils.vis_tool import Visualizer
from utils.config import opt
# import torch.nn.functional as F
from torch.autograd import Variable
from torchnet.meter import ConfusionMeter, AverageValueMeter
if opt.is_distilltion == True:
    LossTuple = namedtuple('LossTuple',
                           ['rpn_loc_loss',
                            'rpn_cls_loss',
                            'roi_loc_loss',
                            'roi_cls_loss',
                            'hint_loss',
                            'total_loss'
                            ])
else:
    LossTuple = namedtuple('LossTuple',
                           ['rpn_loc_loss',
                            'rpn_cls_loss',
                            'roi_loc_loss',
                            'roi_cls_loss',
                            'total_loss'
                            ])


class FocalLoss(nn.Module):
    '''
    test:
        m = nn.Conv2d(16, 4, (3, 3)).float()
        # input is of size N x C x height x width
        input = autograd.Variable(torch.randn(3, 16, 10, 10))
        # each element in target has to have 0 <= value < C
        target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
        loss = FocalLoss(gamma=2,alpha=1)
        output = loss(m(input), target)
        print(output)
        loss = FocalLoss(gamma=2,alpha=[1,1,1,1])
        output = loss(m(input), target)
        print(output)
    '''

    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C

        N = input.size(0)
        C = input.size(1)

        if self.alpha is not None:
            # isinstance mean:判断alpha是否为后面的类型
            if isinstance(self.alpha, (float, int)):
                list_tmp = []
                for n in range(N):
                    list_alphas = []
                    for c in range(C):
                        list_alphas.append(self.alpha)
                        # else:
                        #     list_alphas.append(1 - self.alpha)
                    list_tmp.append(list_alphas)
                self.alpha = t.Tensor(list_tmp)
                if self.alpha.type() != input.data.type():  # convert to cuda
                    self.alpha = self.alpha.type_as(input.data)
                self.alpha = Variable(self.alpha)

        target = target.view(-1, 1)  # N*H*W
        logpt = F.log_softmax(input)  # N*H*W,C

        # gather(input, dim, index,)获得dim维中索引为index的值
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = nn.Softmax()(input)  # N*H*W,C
        pt = pt.gather(1, target).view(-1)
        if self.alpha is not None:
            at = self.alpha.gather(1, target).view(-1)
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


FocalLoss_ = FocalLoss()


class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets.
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter()
                       for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale, epoch,
                teacher_pred_bboxes_, teacher_pred_labels_, teacher_pred_features_):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)
        features = self.faster_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois
        #cat 软标签和真值标签
        if len(teacher_pred_bboxes_) != 0 and opt.is_distilltion == True:
            bbox = t.cat((bbox, teacher_pred_bboxes_))
            label = t.cat((label, teacher_pred_labels_))

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois,
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)

        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # if len(teacher_pred_bboxes_) != 0 and opt.is_distilltion == True:
        #     teacher_sample_roi, teacher_pred_bboxes, teacher_pred_labels = self.proposal_target_creator(
        #         roi,
        #         at.tonumpy(teacher_pred_bboxes_),
        #         at.tonumpy(teacher_pred_labels_),
        #         self.loc_normalize_mean,
        #         self.loc_normalize_std)
        #     teacher_sample_roi_index = t.zeros(len(teacher_sample_roi))
        #     teacher_roi_cls_loc, teacher_roi_score = self.faster_rcnn.head(
        #         features,
        #         teacher_sample_roi,
        #         teacher_sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)

        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(
            rpn_score, gt_rpn_label.cuda(), ignore_index=-1)

        # if len(teacher_pred_bboxes_) != 0 and opt.is_distilltion == True:
        #     teacher_rpn_loc, teacher_rpn_label = self.anchor_target_creator(
        #         at.tonumpy(teacher_pred_bboxes_),
        #         anchor,
        #         img_size)
        #     teacher_rpn_label = at.totensor(teacher_rpn_label).long()
        #     teacher_rpn_loc = at.totensor(teacher_rpn_loc)
        #     teacher_rpn_cls_loss = F.cross_entropy(
        #         rpn_score, teacher_rpn_label.cuda(), ignore_index=-1, reduce=False)
        #     # teacher_rpn_cls_loss = teacher_rpn_cls_loss[1]
        #     # rpn_cls_loss=  teacher_rpn_cls_loss+ rpn_cls_loss
        #     teacher_rpn_loc_loss = _fast_rcnn_loc_loss(
        #         rpn_loc,
        #         teacher_rpn_loc,
        #         teacher_rpn_label.data,
        #         self.rpn_sigma)
        #
        #
        #     # teacher_rpn_loc_loss=loc_l2_loss(teacher_rpn_loc,rpn_loc)
        #     # rpn_loc_loss=teacher_rpn_loc_loss+rpn_loc_loss

        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False),
                        _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(),
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

    #    roi_cls_loss = FocalLoss_(roi_score, gt_roi_label.cuda())
        roi_cls_loss = nn.CrossEntropyLoss()(
            roi_score, gt_roi_label.cuda())  # +roi_cls_loss

        self.roi_cm.add(at.totensor(roi_score, False),
                        gt_roi_label.data.long())

        if opt.is_distilltion == True:
            #hint loss
            hint_loss = l2_loss(features, teacher_pred_features_)
            losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss,
                      roi_cls_loss,
                      hint_loss
                      ]
        else:
            losses = [rpn_loc_loss, rpn_cls_loss,
                      roi_loc_loss,
                      roi_cls_loss
                      ]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    # 训练
    def train_step(self, imgs, bboxes, labels, scale, epoch, teacher_pred_bboxes_=(), teacher_pred_labels_=(),  teacher_pred_features_=()):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale, epoch,
                              teacher_pred_bboxes_, teacher_pred_labels_, teacher_pred_features_)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    # 加载模型
    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        # print state_dict
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
            print('load own model:', path)
            # print 1
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    # 更新混淆矩阵
    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    # 重置混淆矩阵
    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


def l2_loss(gt, pred):
    B, C, H, W = gt.size()
    # loss = t.sum(t.abs(gt - pred))
    loss = t.sum((gt - pred) * (gt - pred)) / (B * C * H * W)
    return loss


def loc_l2_loss(gt, pred):
    H, W = gt.size()
    # loss = t.sum(t.abs(gt - pred))
    loss = t.sum((gt - pred) * (gt - pred)) / (H * W)
    return loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation,
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    # ignore gt_label==-1 for rpn_loss
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss
