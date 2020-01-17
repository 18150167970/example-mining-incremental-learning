#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 这个头文件指的是精准的除法，就是会帮你把小数后面的数值都保留下来，不会去除。
from __future__ import division
from trainer import _fast_rcnn_loc_loss
from utils.draw import draw_predict
from tqdm import tqdm
from torch.utils import data as data_
from data.dataset import Dataset, TestDataset, inverse_normalize, TestDataset_all
from model.utils.bbox_tools import bbox_iou
from collections import defaultdict
from utils import array_tool as at
from utils.vis_tool import vis_bbox
from data.util import read_image
from trainer import FasterRCNNTrainer
from model import FasterRCNNVGG16
from utils.config import opt
from torch import nn
import math
import itertools
import six
import cv2
import numpy as np
import torch as t
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# file='/media/chenli/F1/cell_data/BCCD_Dataset/BCCD/ImageSets/Main/'
file = '/media/chenli/E/VOCdevkit/VOC2007/ImageSets/Main/'
model_path = ['fasterrcnn_12201310_20', 'fasterrcnn_07282152_15', 'fasterrcnn_07290007_15',
              'fasterrcnn_07290223_15', 'fasterrcnn_07290438_15',
              'fasterrcnn_07290653_15', 'fasterrcnn_07290907_15', 'fasterrcnn_07291122_15',
              'fasterrcnn_07291338_15', 'fasterrcnn_07291553_15']
# # 最后五个
# model_path = ['fasterrcnn_01111000_20', 'fasterrcnn_01111921_20', 'fasterrcnn_01120001_20',
#               'fasterrcnn_01120447_20', 'fasterrcnn_01120730_20']

# 随机采样
# model_path = ['fasterrcnn_12201310_20',  'fasterrcnn_01101958_20', 'fasterrcnn_01110519_20',
#               'fasterrcnn_01111921_20', 'fasterrcnn_01120730_20']

# best
# model_path = ['fasterrcnn_12201310_20', 'fasterrcnn_01101517_20',
#               'fasterrcnn_01120730_20', 'fasterrcnn_01120001_20', 'fasterrcnn_01110519_20']

model_score = [0.499, 0.543, 0.578, 0.600, 0.600,
               0.611, 0.623, 0.630, 0.641, 0.656]


opt.caffe_pretrain = True  # this model was trained from torchvision-pretrained model

faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn).cuda()

testset = TestDataset(opt, 'test')
test_dataloader = data_.DataLoader(testset,
                                   batch_size=1,
                                   num_workers=opt.test_num_workers,
                                   shuffle=False,
                                   pin_memory=True
                                   )


def py_soft_nms(dets, socres, label, sigma=0.5, Nt=0.3, threshold=0.001):
    # print threshold
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = socres  # bbox打分

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # print xx1.shape

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #         print ovr
        inds = []
        #         print label[0]
        b = order[0]
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        # 只有在标签一致的情况下才会判断是否被窗口吸收，如果没有标签的话就使用下面注释的代码
        for j in range(len(order) - 1):
            a = order[j + 1]
            if opt.nms_use_label:
                if label[a] == label[b]:
                    if ovr[j] > Nt:
                        scores[a] = scores[a] * (1 - ovr[j])
                        if scores[a] > threshold:
                            inds.append(j + 1)

                    else:
                        inds.append(j + 1)
                else:
                    inds.append(j + 1)
            else:
                if ovr[j] > Nt:
                    scores[a] = scores[a] * (1 - ovr[j])
                    if scores[a] > threshold:
                        inds.append(j + 1)

                else:
                    inds.append(j + 1)
        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        order = order[inds]

    return keep


# nms 非极大值抑制
# 输入为bbox数据,包含[ymin,xmin,ymax,xmax];每个bbox的评分；每个bbox的标签，如果没有标签就注释掉
def py_cpu_nms(dets, socres, label, thresh=0.0000001, threshold=0.01):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = socres  # bbox打分
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 打分从大到小排列，取index
    order = scores.argsort()[::-1]
    # keep为最后保留的边框
    keep = []
    while order.size > 0:
        # order[0]是当前分数最大的窗口，肯定保留
        i = order[0]
        keep.append(i)
        # 计算窗口i与其他所有窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        #         print xx1.shape

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 交/并得到iou值
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #         print ovr
        # inds = []
        # #         print label[0]
        # b = order[0]
        # inds为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        # 只有在标签一致的情况下才会判断是否被窗口吸收，如果没有标签的话就使用下面注释的代码
        # for j in range(len(order) - 1):
        #     a = order[j + 1]
        # if opt.nms_use_label:
        #     if label[a] == label[b]:
        #         if ovr[j] <= thresh:
        #             inds.append(j + 1)
        #     else:
        #         inds.append(j + 1)
        # else:
        # if ovr[j] <= thresh:
        #     inds.append(j + 1)

        # order里面只保留与窗口i交叠面积小于threshold的那些窗口，由于ovr长度比order长度少1(不包含i)，所以inds+1对应到保留的窗口
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):

    prec, rec = calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)
    #     print prec
    #     print prec,rec
    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_detection_voc_ap(prec, rec, use_07_metric=False):

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):
        if prec[l] is None or rec[l] is None:
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec[l] >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))
            mrec = np.concatenate(([0], rec[l], [1]))

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

# 计算的召回率和准确率，每一种都包含类别个数大小的数threshold组，每一个代表一个类别的召回率或者准确率


def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults=None,
        iou_thresh=0.5):

    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes1 = iter(gt_bboxes)
    gt_labels1 = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults1 = itertools.repeat(None)
    else:
        gt_difficults1 = iter(gt_difficults)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes1, gt_labels1, gt_difficults1):

        #         print pred_bbox
        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
            # 在真实标签中选出标签为某值的boundingbox
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]

            # sort by score 对分数排序
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]

            # 在真实标签中选出标签为某值的boundingbox
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            #             print n_pos[l]
            # list.extend 追加一行
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.VOC评价遵循整数bounding boxes
            pred_bbox_l = pred_bbox_l.copy()
            #             print pred_bbox_l
            pred_bbox_l[:, 2:] += 1
            #             print pred_bbox_l
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1

            # 找到所有gt和pred的重叠面积，总共gt.shape*pred.shape 个重叠面积
            iou = bbox_iou(pred_bbox_l, gt_bbox_l)
            # 找到最大的和真实样本bbox的重叠面积的索引
            gt_index = iou.argmax(axis=1)
            #             print gt_index
            # set -1 if there is no matching ground truth
            # 小于阈值的就去除掉
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            # 计算匹配的个数
            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    if gt_difficult_l[gt_idx]:
                        match[l].append(-1)
                    else:
                        if not selec[gt_idx]:
                            match[l].append(1)
                        else:
                            match[l].append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)

    for iter_ in (
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes1, gt_labels1, gt_difficults1):
        if next(iter_, None) is not None:
            raise ValueError('Length of input iterables need to be same.')

    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)

        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
    return prec, rec

# 传入的是真值标签和预测标签


def bbox_iou(bbox_a, bbox_b):

    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
  # 由于可能两个bbox大小不一致，所以使用[bbox_a,bbox_b,2]存储遍历的bbox_a*bbox_b个bbox的比较
    # top left  这边是计算了如图上第一幅的重叠左下角坐标值（x，y）
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right  这边是计算了如图上第一幅的重叠左上角坐标值ymax和右下角坐标值xmax

    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    # np.prod 给定轴数值的乘积   相减就得到高和宽 然后相乘
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)  # 重叠部分面积
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    return area_i / (area_a[:, None] + area_b - area_i)


def bbox_result(dataloader, faster_rcnn, test_num=opt.ensemble_test_numebr):
    # 获得预测结果
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults, ID = list(), list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]

        pred_bboxes_, pred_labels_, pred_scores_, _ = faster_rcnn.predict(imgs, [
            sizes])
        # print(ii)
        # print gt_bboxes_
        np.save(
            'bboxs/' + str(id_[0]) + '.npy',pred_bboxes_)
        np.save(
            'labels/' + str(id_[0]) + '.npy',pred_labels_)
        np.save(
            'scores/' + str(id_[0]) + '.npy',pred_scores_)
        np.save(
            'difficult/' + str(id_[0]) + '.npy',gt_difficults_.numpy())
        np.save(
            'gt_box/' + str(id_[0]) + '.npy',gt_bboxes_.numpy())
        np.save(
            'gt_label/' + str(id_[0]) + '.npy',gt_labels_.numpy())
        # f.write(str(id_[0])+'\n')
        ID += list(id_)
        if ii == test_num:
            break

    for i in range(len(ID)):
        id_=ID[i]
        if len(id_)==0:
            break
        pred_bboxes_= np.load(
            'bboxs/' + str(id_) + '.npy')
        pred_labels_ = np.load(
            'labels/' + str(id_) + '.npy')
        pred_scores_ = np.load(
            'scores/' + str(id_) + '.npy')
        gt_bboxes_ = np.load(
            'gt_box/' + str(id_) + '.npy')
        gt_labels_ = np.load(
            'gt_label/' + str(id_) + '.npy')
        gt_difficults_ = np.load(
            'difficult/' + str(id_) + '.npy')
        gt_bboxes += list(gt_bboxes_)
        gt_labels += list(gt_labels_)
        gt_difficults += list(gt_difficults_)
        pred_bboxes += list(pred_bboxes_)
        pred_labels += list(pred_labels_)
        pred_scores += list(pred_scores_)


    return pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults, ID


def nms_reuslt(
        pred_bboxes, pred_labels, pred_scores, pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes, gt_labels, gt_difficults, threshold=0.001):
    # 计算的召回率和准确率，每一种都包含类别个数大小的数threshold组，每一个代表一个类别的召回率或者准确率
    pred_bboxes1 = iter(pred_bboxes)
    pred_labels1 = iter(pred_labels)
    pred_scores1 = iter(pred_scores)
    pred_bboxes_1 = iter(pred_bboxes_)
    pred_labels_1 = iter(pred_labels_)
    pred_scores_1 = iter(pred_scores_)

    pred_b, pred_l, pred_s = list(), list(), list()

    for pred_bbox, pred_label, pred_score, pred_bbox_, pred_label_, pred_score_ in \
            six.moves.zip(
                pred_bboxes1, pred_labels1, pred_scores1,
                pred_bboxes_1, pred_labels_1, pred_scores_1):

        bounding_boxes = np.concatenate((pred_bbox, pred_bbox_))
        confidence_score = np.concatenate((pred_score, pred_score_))
        labels = np.concatenate((pred_label, pred_label_))
        if opt.nms_type == 'soft_nms':
            keep = py_soft_nms(bounding_boxes, confidence_score,
                               labels, threshold=threshold)
            # print opt.nms_type
        else:
            keep = py_cpu_nms(bounding_boxes, confidence_score,
                              labels, threshold=threshold)
            # print opt.nms_type
        bounding = bounding_boxes[keep]
        confidence = confidence_score[keep]
        label = labels[keep]

        bbox1 = np.expand_dims(bounding, axis=0)
        label1 = np.expand_dims(label, axis=0)
        confidence1 = np.expand_dims(confidence, axis=0)

        pred_b += list(bbox1)
        pred_l += list(label1)
        pred_s += list(confidence1)

    result = eval_detection_voc(
        pred_b, pred_l, pred_s, gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    result2 = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    result3 = eval_detection_voc(
        pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    # print(result, result2, result3)
    if result['map'] < result2['map']:
        result = result2
        pred_b, pred_l, pred_s = pred_bboxes, pred_labels, pred_scores
    if result['map'] < result3['map']:
        result = result3
        pred_b, pred_l, pred_s = pred_bboxes_, pred_labels_, pred_scores_
    return pred_b, pred_l, pred_s, result


def concat_reuslt(
        pred_bboxes, pred_labels, pred_scores, pred_bboxes_, pred_labels_, pred_scores_):
    # 合并两个预测结果为一个
    pred_bboxes1 = iter(pred_bboxes)
    pred_labels1 = iter(pred_labels)
    pred_scores1 = iter(pred_scores)
    pred_bboxes_1 = iter(pred_bboxes_)
    pred_labels_1 = iter(pred_labels_)
    pred_scores_1 = iter(pred_scores_)

    pred_b, pred_l, pred_s = list(), list(), list()

    for pred_bbox, pred_label, pred_score, pred_bbox_, pred_label_, pred_score_ in \
            six.moves.zip(
                pred_bboxes1, pred_labels1, pred_scores1,
                pred_bboxes_1, pred_labels_1, pred_scores_1):
        # print pred_bbox.shape
        # print pred_bbox_.shape
        bounding_boxes = np.concatenate((pred_bbox, pred_bbox_))
        # print ('concat', bounding_boxes.shape)
        confidence_score = np.concatenate((pred_score, pred_score_))
        labels = np.concatenate((pred_label, pred_label_))

        bbox1 = np.expand_dims(bounding_boxes, axis=0)
        label1 = np.expand_dims(labels, axis=0)
        confidence1 = np.expand_dims(confidence_score, axis=0)
        pred_b += list(bbox1)
        pred_l += list(label1)
        pred_s += list(confidence1)
    return pred_b, pred_l, pred_s


def nms_five_reuslt(
    pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults, threshold=0.001, nms_type='soft_nms'):
    pred_bboxes1 = iter(pred_bboxes)
    pred_labels1 = iter(pred_labels)
    pred_scores1 = iter(pred_scores)

    pred_b, pred_l, pred_s = list(), list(), list()

    for pred_bbox, pred_label, pred_score in six.moves.zip(pred_bboxes1, pred_labels1, pred_scores1):
        if nms_type == 'soft_nms':
            # print threshold
            keep = py_soft_nms(pred_bbox, pred_score,
                               pred_label, threshold=threshold)

        else:
            keep = py_cpu_nms(pred_bbox, pred_score, pred_label)

        bounding = pred_bbox[keep]
        confidence = pred_score[keep]
        label = pred_label[keep]

        bbox1 = np.expand_dims(bounding, axis=0)
        label1 = np.expand_dims(label, axis=0)
        confidence1 = np.expand_dims(confidence, axis=0)

        pred_b += list(bbox1)
        pred_l += list(label1)
        pred_s += list(confidence1)

    result = eval_detection_voc(
        pred_b, pred_l, pred_s, gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    return pred_b, pred_l, pred_s, result


def concat_five_result():
    # 合并五个预测结果,然后nms结果,最后评测# trainer.load('checkpoints/' + model_path[0])
    trainer.load('checkpoints/' + model_path[5])
    pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
        test_dataloader, trainer.faster_rcnn)

    print(pred_bboxes_[0][pred_scores_[0]>0.98])
    for i in range(6, 10):
        trainer.load('checkpoints/' + model_path[i])
        pred_bboxes1, pred_labels1, pred_scores1, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
            test_dataloader, trainer.faster_rcnn)

        # print(pred_scores1)
        pred_bboxes_, pred_labels_, pred_scores_ = concat_reuslt(pred_bboxes1, pred_labels1, pred_scores1,
                                                                 pred_bboxes_, pred_labels_, pred_scores_)
        pred_bboxes1=pred_bboxes1[0]
        pred_bboxes1=pred_bboxes1[pred_scores1[0]>0.98]
        print(pred_bboxes1)
    threshold=0.001

    # draw_predict(pred_bboxes_, pred_labels_, pred_scores_)

    pred_labels_=pred_labels_[0].reshape(-1)
    pred_bboxes_=pred_bboxes_[0]
    pred_scores_=pred_scores_[0]
    print(pred_labels_.shape)
    print(pred_bboxes_.shape)
    print(pred_scores_.shape)
    # print(pred_labels_==11)
    # pred_labels_=pred_labels_[pred_labels_==11]
    # pred_bboxes_=pred_bboxes_[pred_labels_==11]
    from data.util import read_image
    from utils.vis_tool import visdom_bbox
    img_file = os.path.join('/media/chenli/E/VOCdevkit/VOC2007/JPEGImages', '001150' + '.jpg')
    img = read_image(img_file, color=True)
    pred_bboxes_=pred_bboxes_[pred_labels_==11]
    pred_scores_=pred_scores_[pred_labels_==11]
    pred_labels_=pred_labels_[pred_labels_==11]

    pred_img = visdom_bbox(img,
                            pred_bboxes_[pred_scores_>0.7],
                           pred_labels_[pred_scores_>0.7],
                           )

    trainer.vis.img('nms_five', pred_img)
    # # for j in range(1, 20):
    #     print pred_bboxes_[0].shape
    #     threshold = (0.001 + 0.001 * j)
    #     pred_b, pred_l, pred_s, result = nms_five_reuslt(
    #         pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes, gt_labels, gt_difficults, nms_type='soft_nms', threshold=threshold)
    #     # print pred_b[0].shape
    #     print('threshold', threshold, 'soft_nms result', result['map'])

    # pred_b, pred_l, pred_s, result = nms_five_reuslt(
    #     pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes, gt_labels, gt_difficults, nms_type='nms', threshold=threshold)
    # # print pred_b[0].shape
    # print('nms', result['map'])
    # print('nms result five completed')
    return 0

# concat_five_result()

def draw_plt(x, y):
    # 绘制NMS阈值曲线图
    plt.plot(x, y, label='soft-NMS-label')
    plt.xlabel('threshold')
    plt.ylabel('mAP(%)')
    plt.title("soft-NMS-label threshold")
    plt.legend()
    # plt.show()
    plt.savefig('soft-NMS-label.png', dpi=360)


def one_by_one(ensemble_number = 5):
    if opt.nms_type == 'soft_nms':
        epoch_number = 1
    else:
        epoch_number = 1
    x = list()
    y = list()
    gt_bboxes = np.load('bbox/' + 'gt.npy')
    gt_labels = np.load('label/' + 'gt.npy')
    gt_difficults = np.load('score/' + 'gt.npy')
    for j in range(0, epoch_number):
        pred_bboxes_ = np.load('bbox/' + model_path[0] + '.npy')
        pred_labels_ = np.load('label/' + model_path[0] + '.npy')
        pred_scores_ = np.load('score/' + model_path[0] + '.npy')
        # threshold = (0.00001 + 0.0001 * j)
        threshold = 0.0001
        for i in range(1, ensemble_number):
            print(i)
            pred_bboxes1 = np.load('bbox/' + model_path[i] + '.npy')
            pred_labels1 = np.load('label/' + model_path[i] + '.npy')
            pred_scores1 = np.load('score/' + model_path[i] + '.npy')
            pred_bboxes_, pred_labels_, pred_scores_, result = nms_reuslt(pred_bboxes1, pred_labels1, pred_scores1,
                                                                          pred_bboxes_, pred_labels_, pred_scores_,
                                                                          gt_bboxes, gt_labels, gt_difficults, threshold=threshold)
            # print pred_bboxes_[0].shape
        print('threshold', threshold, 'result', result)
        x.append(threshold)
        y.append(result)

    # draw_plt(x, y)
    # draw_predict(pred_bboxes_, pred_labels_, pred_scores_)
    print('nms result one completed')
    return x, y


def diversity_map(pred_bboxes, pred_labels, pred_scores, gt_bboxes1, gt_labels1, gt_difficult1):
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes1, gt_labels1,
        use_07_metric=True)
    return result['map']


def cross_entropy(a, y):
    return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))


def bbox_iou2(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.max(b1_x1, b2_x1)
    inter_rect_y1 = np.max(b1_y1, b2_y1)
    inter_rect_x2 = np.min(b1_x2, b2_x2)
    inter_rect_y2 = np.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * np.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def build_targets(target_bbox, target_labels, target_scores, anchors, num_anchors, num_classes, nX, nY, ignore_thres):
    #
    nB = 1
    nA = num_anchors  # 锚点数
    nC = num_classes  # 类别数
    # print target_bbox.shape
    # print nX, nY
    # print target_bbox
    mask = np.zeros((nB, nA, nX, nY))  # (batch_size,3,13/26/52,13/26/52)
    conf_mask = np.ones((nB, nA, nX, nY))
    tx = np.zeros((nB, nA, nX, nY))
    ty = np.zeros((nB, nA, nX, nY))
    tw = np.zeros((nB, nA, nX, nY))
    th = np.zeros((nB, nA, nX, nY))
    tconf = np.zeros((nB, nA, nX, nY)).astype('float32')
    tcls = np.zeros((nB, nA, nX, nY, nC)).astype('float32')
    b = 0
    for t in range(target_bbox.shape[0]):
        # if target_bbox[b, t].sum() == 0:
        #     continue

        # target存储相对坐标,所以还原需要乘上特征图大小
        gx = target_bbox[t, 0]
        gy = target_bbox[t, 1]
        gw = target_bbox[t, 2] - target_bbox[t, 0]
        gh = target_bbox[t, 3] - target_bbox[t, 1]
        gi = int(gx)  # 网格坐标
        # print gw
        gj = int(gy)
        gt_box = np.array([0, 0, int(gw), int(gh)])
        gt_box = gt_box[np.newaxis, :]

        # Get shape of anchor box
        anchor_shapes = np.concatenate(
            (np.zeros((len(anchors), 2)), np.array(anchors)), 1)
        # Calculate iou between gt and anchor shapes
        # print gt_box.shape
        # print anchor_shapes.shape
        anch_ious = bbox_iou(gt_box, anchor_shapes)
        # print anch_ious.shape
        # Where the overlap is larger than threshold set mask to zero (ignore)
        # print conf_mask.shape
        conf_mask[b, anch_ious[0] > ignore_thres, gi, gj] = 0
        # Find the best matching anchor box
        best_n = np.argmax(anch_ious)
        # Get ground truth box
        gt_box = np.array([gx, gy, gw, gh])
        gt_box = gt_box[np.newaxis, :]

        # Masks,用于找到最高重叠率的预测窗口
        mask[b, best_n,  gi, gj] = 1
        conf_mask[b, best_n,  gi, gj] = 1
        # 真值标签相对网格点坐标
        tx[b, best_n,  gi, gj] = gx - gi
        ty[b, best_n,  gi, gj] = gy - gj
        # Width and height
        tw[b, best_n,  gi, gj] = math.log(gw / anchors[best_n][0] + 1e-16)
        th[b, best_n,  gi, gj] = math.log(gh / anchors[best_n][1] + 1e-16)
        # One-hot encoding of label
        # print target_labels
        target_label = target_labels
        tcls[b, best_n,  gi, gj, target_label] = 1
        tconf[b, best_n,  gi, gj] = 1

    return mask, conf_mask, tx, ty, tw, th, tconf, tcls


def mse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def cross_entropy(a, y):
    return -(np.dot(y.transpose(), np.nan_to_num(np.log(a))) + np.dot((1 - y).transpose(), np.nan_to_num(np.log(1 - a))))


def diversity_loss(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults):
    different = 0
    anchors = [(10, 13), (16, 30), (33, 23)]
    loss = 0
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in tqdm(enumerate(test_dataloader)):
        nA = 3  # 3
        pred_bbox, pred_label, pred_score, = pred_bboxes[ii], pred_labels[ii], pred_scores[ii]
        gt_bboxe, gt_label, gt_difficult = gt_bboxes[ii], gt_labels[ii], gt_difficults[ii]
        nB = imgs.size(0)  # 1
        nX = imgs.size(2)
        nY = imgs.size(3)

        scaled_anchors = [(a_w, a_h) for a_w, a_h in anchors]
        mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
            target_bbox=gt_bboxe,
            target_labels=gt_label,
            target_scores=gt_difficult,
            anchors=scaled_anchors,
            num_anchors=nA,
            num_classes=len(opt.VOC_BBOX_LABEL_NAMES),
            nX=nX,
            nY=nY,
            ignore_thres=0.5,
        )
        mask2, conf_mask2, x, y, w, h, pred_conf, pred_cls = build_targets(
            target_bbox=pred_bbox,
            target_labels=pred_label,
            target_scores=pred_score,
            anchors=scaled_anchors,
            num_anchors=nA,
            num_classes=len(opt.VOC_BBOX_LABEL_NAMES),
            nX=nX,
            nY=nY,
            ignore_thres=0.5,
        )
        mask = mask.astype('int64')
        mask2 = mask2.astype('int64')
        conf_mask_true = mask
        conf_mask_false = conf_mask - mask
        # print x[mask2]
        # print x.shape
        # print tx.shape
        loss_x = mse(x, tx)
        loss_y = mse(y, ty)
        loss_w = mse(w, tw)
        loss_h = mse(h, th)
        # loss_conf = nn.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + nn.bce_loss(
        #     pred_conf[conf_mask_true], tconf[conf_mask_true]
        # )
        # print np.argmax(pred_cls, 1).shape
        # print np.argmax(tcls, 1).shape
        # print pred_cls.shape
        # print mask.shape
        pred_cls = np.argmax(pred_cls, 4)
        pred_cls = pred_cls[mask == 1]
        tcls = np.argmax(tcls, 4)
        tcls = tcls[mask == 1]
        # print pred_cls, tcls
        loss_cls = cross_entropy(
            tcls, pred_cls)
        loss += loss_x + loss_y + loss_w + loss_h
        # print loss
    return loss


def different_ensemble(model_path_, ensemble_type='loss', thread=0.5):
    # 去掉一个模型
    pred_bboxes_, pred_labels_, pred_scores_ = list(), list(), list()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes = np.load('bbox/' + 'gt.npy')
    gt_labels = np.load('label/' + 'gt.npy')
    gt_difficults = np.load('score/' + 'gt.npy')

    total_different = 1
    for i in range(6):
        pred_bboxes1 = np.load('bbox/' + model_path_[i] + '.npy')
        pred_labels1 = np.load('label/' + model_path_[i] + '.npy')
        pred_scores1 = np.load('score/' + model_path_[i] + '.npy')

        pred_bboxes_.append(pred_bboxes1)
        pred_labels_.append(pred_labels1)
        pred_scores_.append(pred_scores1)

    # 找到多样性最差的一个,也就是和别的模型结果最相似的那个,越相似map越高
    a = np.zeros((6, 6))
    max_number = 0
    max_number_index = 0
    sum_different = 0
    for i in range(6):
        for j in range(i + 1, 6):
            if len(pred_bboxes_[i][0]) == 0 or len(pred_bboxes_[j][0]) == 0:
                total_different += 1
                continue
            # print(pred_bboxes_[i].shape, pred_bboxes_[j].shape)
            if ensemble_type == 'mAP':
                maps = diversity_map(pred_bboxes_[i], pred_labels_[i], pred_scores_[
                    i], pred_bboxes_[j], pred_labels_[j], pred_scores_[j])
                a[i, j] = maps
                a[j, i] = maps
            else:
                maps = diversity_loss(pred_bboxes_[i], pred_labels_[i], pred_scores_[
                    i], pred_bboxes_[j], pred_labels_[j], pred_scores_[j])
                a[i, j] = maps
                a[j, i] = maps
        if opt.ensemble_use_map:
            if ensemble_type == 'mAP':
                total_score = a[i].sum() - model_score[i]
            else:
                total_score = 1 - (a[i].sum() / 800.0) - model_score[i]
        else:
            if ensemble_type == 'mAP':
                total_score = a[i].sum()
            else:
                total_score = 1 - (a[i].sum() / 800.0)
        sum_different += total_score
        if total_score > max_number:
            max_number = total_score
            max_number_index = i
    # 用新增的模型替换旧模型
    model_path_[max_number_index] = model_path_[5]
    model_score[max_number_index] = model_score[5]
    f=open("result.txt","a")
    f.write("similarly:"+str(sum_different-2*max_number)+'\n')

    f.write("max_number_index:"+str(max_number_index)+'\n')
    f.close()
    print(sum_different-max_number)
    print(max_number_index)
    return model_path_[0:5], max_number_index

def different_ensemble2(model_path_, ensemble_type='loss', thread=0.5):
    # 去掉一个模型
    pred_bboxes_, pred_labels_, pred_scores_ = list(), list(), list()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes = np.load('bbox/' + 'gt.npy')
    gt_labels = np.load('label/' + 'gt.npy')
    gt_difficults = np.load('score/' + 'gt.npy')

    total_different = 1
    for i in range(6):
        pred_bboxes1 = np.load('bbox/' + model_path[i] + '.npy')
        pred_labels1 = np.load('label/' + model_path[i] + '.npy')
        pred_scores1 = np.load('score/' + model_path[i] + '.npy')

        pred_bboxes_.append(pred_bboxes1)
        pred_labels_.append(pred_labels1)
        pred_scores_.append(pred_scores1)

    # 找到多样性最差的一个,也就是和别的模型结果最相似的那个,越相似map越高
    a = np.zeros((6, 6))
    max_number = 0
    max_number_index = 0
    sum_different = 0
    for i in range(6):
        for j in range(i + 1, 6):
            if len(pred_bboxes_[i][0]) == 0 or len(pred_bboxes_[j][0]) == 0:
                total_different += 1
                continue
            # print(pred_bboxes_[i].shape, pred_bboxes_[j].shape)
            if ensemble_type == 'mAP':
                maps = diversity_map(pred_bboxes_[i], pred_labels_[i], pred_scores_[
                    i], pred_bboxes_[j], pred_labels_[j], pred_scores_[j])
                a[i, j] = maps
                a[j, i] = maps
            else:
                maps = diversity_loss(pred_bboxes_[i], pred_labels_[i], pred_scores_[
                    i], pred_bboxes_[j], pred_labels_[j], pred_scores_[j])
                a[i, j] = maps
                a[j, i] = maps
        if opt.ensemble_use_map:
            if ensemble_type == 'mAP':
                total_score = a[i].sum() - model_score[i]
            else:
                total_score = 1 - (a[i].sum() / 800.0) - model_score[i]
        else:
            if ensemble_type == 'mAP':
                total_score = a[i].sum()
            else:
                total_score = 1 - (a[i].sum() / 800.0)
        sum_different += total_score

    f=open("result.txt","a")
    f.write("similarly:"+str(sum_different-2*(1 - (a[0].sum() / 800.0)))+'\n')
    f.close()


    return model_path_[1:6], max_number_index

# for i in range(0, 10):
#     trainer.load('checkpoints/' + model_path[i])
#     pred_bboxes1, pred_labels1, pred_scores1, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
#         test_dataloader, trainer.faster_rcnn)
#     np.save('bbox/' + model_path[i] + '.npy', pred_bboxes1)
#     np.save('label/' + model_path[i] + '.npy', pred_labels1)
#     np.save('score/' + model_path[i] + '.npy', pred_scores1)
#     np.save('bbox/' + 'gt.npy', gt_bboxes)
#     np.save('label/' + 'gt.npy', gt_labels)
#     np.save('score/' + 'gt.npy', gt_difficults)


# model_path = ['fasterrcnn_12201310_20', 'fasterrcnn_01111000_20', 'fasterrcnn_01101958_20',
#               'fasterrcnn_01110039_20', 'fasterrcnn_01110519_20', 'fasterrcnn_01120730_20']

# f=open("result.txt","a")
# f.write("ensemble_type:"+opt.ensemble_type+'\n')
# for i in range(5, 10):
#     model_path[5] = model_path[i]
#     model_score[5] = model_score[i]
#     print(model_path[0:6])
#     f.write(str(model_path[0:6])+'\n')
#     model_path[0:5], max_number_index = different_ensemble(model_path[0:6],ensemble_type = opt.ensemble_type)
#
#     print('ensembel completed', i)
#     # model_path[i - 5] = model_path[i]
#     x,y=one_by_one()
#     f.write("result:"+str(y)+'\n')
# f.close()


f=open("result.txt","a")
f.write("ensemble_type:"+opt.ensemble_type+'\n')
for i in range(6, 11):
    # model_path[0:5], max_number_index = different_ensemble(model_path[0:i],ensemble_type = opt.ensemble_type)

    print('ensembel completed', i)
    # model_path[i - 5] = model_path[i]
    x,y=one_by_one(i)
    f.write("result:"+str(y)+'\n')
f.close()

# print(model_path[0:5])
# print('ensembel completed')

# opt.nms_type = 'nms'  # soft_nms nms
# opt.nms_use_label = False
# xx, yy = one_by_one()
# plt.plot(xx, yy, label='NMS')
#
# opt.nms_type = 'nms'  # soft_nms nms
# opt.nms_use_label = True
# xx, yy = one_by_one()
# plt.plot(xx, yy, label='NMS-label')
#
# opt.nms_type = 'soft_nms'  # soft_nms nms
# opt.nms_use_label = False
# xx, yy = one_by_one()
# plt.plot(xx, yy, label='soft-NMS')
#
# opt.nms_type = 'soft_nms'  # soft_nms nms
# opt.nms_use_label = True
# xx, yy = one_by_one()
# plt.plot(xx, yy, label='soft-NMS-label')
#
# plt.xlabel('incremental number')
# plt.ylabel('mAP(%)')
# plt.title("NMS-class")
# plt.legend()
# plt.savefig('NMS-class.png', dpi=360)
