#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 这个头文件指的是精准的除法，就是会帮你把小数后面的数值都保留下来，不会去除。
from __future__ import division
import os
import torch as t
import numpy as np
import cv2
import six
import itertools

from utils.config import opt
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
from collections import defaultdict
from model.utils.bbox_tools import bbox_iou
from data.dataset import Dataset, TestDataset, inverse_normalize
from torch.utils import data as data_
from tqdm import tqdm
from utils.eval_tool import eval_detection_voc


def example_mining_map(trainer, modify_txt_path):
    # 加载权重
    trainset = TestDataset(opt, split=str(int(modify_txt_path) - 1))
    train_dataloader = data_.DataLoader(trainset,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False,
                                        pin_memory=True
                                        )

    pred_bboxes1, pred_labels1, pred_scores1, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
        train_dataloader, trainer.faster_rcnn, test_num=1100)
    map_result = every_map(pred_bboxes1, pred_labels1,
                           pred_scores1, gt_bboxes, gt_labels, gt_difficults)
    # print map_result
    modify(modify_txt_path, map_result, ID)


def example_mining_map_diversity(trainer, modify_txt_path):
    # 加载权重
    trainset = TestDataset(opt, split=str(int(modify_txt_path) - 1))
    train_dataloader = data_.DataLoader(trainset,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False,
                                        pin_memory=True
                                        )

    pred_bboxes1, pred_labels1, pred_scores1, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
        train_dataloader, trainer.faster_rcnn, test_num=1100)
    map_result = every_map(pred_bboxes1, pred_labels1,
                           pred_scores1, gt_bboxes, gt_labels, gt_difficults)

    if opt.example_sort == 'max':
        total_different = np.zeros(2000)
    else:
        total_different = np.zeros(2000) + 1000
    ID2 = list()

    trainset = TestDataset(opt, split=str(int(modify_txt_path) - 1))
    train_dataloader = data_.DataLoader(trainset,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False,
                                        pin_memory=True
                                        )

    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in enumerate(train_dataloader):
        if len(gt_bboxes_) == 0:
            continue
        # print('1', imgs.shape)
        ID2 += list(id_)
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        total_different[ii] = different(
            imgs, gt_bboxes_, sizes, trainer.faster_rcnn, thread=0.3)

    if opt.example_sort == 'max':
        order = map_result.argsort()
        order2 = total_different.argsort()[::-1]
    else:
        order = map_result.argsort()[::-1]
        order2 = total_different.argsort()

    sum = 0
    f = open(opt.voc_data_dir + '/ImageSets/Main/' +
             modify_txt_path + '.txt', "a")
    for i in range(500):
        for j in range(opt.mining_number * 2):
            if ID2[order2[i]] == ID[order[j]]:
                f.write(ID2[order2[i]] + '\n')
                sum += 1
            if sum >= opt.mining_number:
                break


def example_mining_diversity(trainer, modify_txt_path):
    if opt.example_sort == 'max':
        total_different = np.zeros(2000)
    else:
        total_different = np.zeros(2000) + 1000
    ID = list()

    trainset = TestDataset(opt, split=str(int(modify_txt_path) - 1))
    train_dataloader = data_.DataLoader(trainset,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False,
                                        pin_memory=True
                                        )

    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in enumerate(train_dataloader):
        if len(gt_bboxes_) == 0:
            continue
        # print('1', imgs.shape)
        ID += list(id_)
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        total_different[ii] = different(
            imgs, gt_bboxes_, sizes, trainer.faster_rcnn, thread=0.7)

    modify(modify_txt_path, total_different, ID)


def example_mining_map_loss(trainer, modify_txt_path):
    # 加载权重
    trainset = TestDataset(opt, split=str(int(modify_txt_path) - 1))
    train_dataloader = data_.DataLoader(trainset,
                                        batch_size=1,
                                        num_workers=opt.test_num_workers,
                                        shuffle=False,
                                        pin_memory=True
                                        )

    pred_bboxes1, pred_labels1, pred_scores1, gt_bboxes, gt_labels, gt_difficults, ID = bbox_result(
        train_dataloader, trainer.faster_rcnn, test_num=1100)
    map_result = every_map(pred_bboxes1, pred_labels1,
                           pred_scores1, gt_bboxes, gt_labels, gt_difficults)

    f = open('loss.txt', "r")
    a = dict.fromkeys(ID)
    for i in range(len(ID)):
        line = f.readline()
        a[line[0:6]] = float(line[7:-2])
    f.close()
    for i in range(len(ID)):
        map_result[i] = a[ID[i]] - map_result[i]
    modify(modify_txt_path, map_result, ID)


def example_mining_loss(datatxt):
    # 加载权重
    f = open('loss.txt', "r")
    f2 = open(opt.voc_data_dir + '/ImageSets/Main/' +
              datatxt + '.txt', "a")
    for i in range(opt.mining_number):
        f2.write(f.readline()[0:6] + '\n')
    f.close()
    f2.close()


def bbox_iou(bbox_a, bbox_b):
    # 传入的是真值标签和预测标签
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


def bbox_result(dataloader, faster_rcnn, test_num=2000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults, ID = list(), list(), list(), list()

    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_, _feature_ = faster_rcnn.predict(imgs, [
            sizes])

        # print gt_bboxes_
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        ID += list(id_)

        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii >= test_num:
            break

    return pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults, ID


def every_map(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes1, gt_labels1, gt_difficults1=None,
        iou_thresh=0.5):

    map_result = np.zeros((len(pred_labels)))
    pred_bboxes = iter(pred_bboxes)
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)

    gt_bboxes1 = iter(gt_bboxes1)
    gt_labels1 = iter(gt_labels1)
    if gt_difficults1 is None:
        gt_difficults1 = itertools.repeat(None)
    else:
        gt_difficults1 = iter(gt_difficults1)

    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    # print map_result.shape
    i = 0
    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes1, gt_labels1, gt_difficults1):

        pred_bboxes_, pred_labels_, pred_scores_ = list(), list(), list()
        gt_bboxes_, gt_labels_, gt_difficults_ = list(), list(), list()
        bbox1 = np.expand_dims(gt_bbox, axis=0)
        label1 = np.expand_dims(pred_label, axis=0)
        labels1 = np.expand_dims(gt_label, axis=0)
        bounding1 = np.expand_dims(pred_bbox, axis=0)
        confidence1 = np.expand_dims(pred_score, axis=0)
        difficults1 = np.expand_dims(gt_difficult, axis=0)

        gt_bboxes_ += list(bbox1)
        gt_labels_ += list(labels1)
        gt_difficults_ += list(difficults1)
        pred_bboxes_ += list(bounding1)
        pred_labels_ += list(label1)
        pred_scores_ += list(confidence1)
        # print pred_bboxes_[0].shape
        result = eval_detection_voc(
            pred_bboxes_, pred_labels_, pred_scores_, gt_bboxes_, gt_labels_, gt_difficults_,
            use_07_metric=True)
        map_result[i] = result['map']
        i += 1
    return map_result


def modify(datapath, map_result, ID):
    if opt.example_sort == 'max':
        order = map_result.argsort()[::-1]
    else:
        order = map_result.argsort()
    f = open(opt.voc_data_dir + '/ImageSets/Main/' + datapath + '.txt', "a")
    for i in range(opt.mining_number):
        f.write(ID[order[i]] + '\n')
    f.close()


def imgflip(img, bbox, x_flip=True, y_flip=True):
    imgs = at.tonumpy(img[0])
    if y_flip:
        imgs = imgs[:, ::-1, :]
    if x_flip:
        imgs = imgs[:, :, ::-1]
    # print imgs
    imgs = np.expand_dims(imgs, axis=0)
    return inverse_normalize(imgs)


def bbox_flip(img, bbox, x_flip=True, y_flip=True):
    H, W = img[0].shape
    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max
    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max
    return bbox


def different(img, bbox, sizes, faster_rcnn, thread=0.5):
    pred_bboxes_, pred_labels_, pred_scores_ = list(), list(), list()
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    total_different = 1

    img_flip = imgflip(img, bbox, x_flip=True, y_flip=True)
    pred_bboxes_1, pred_labels_1, pred_scores_1, _ = faster_rcnn.predict(img_flip, [
        sizes], visualize=True)
    pred_bboxes_1 = bbox_flip(img_flip[0], pred_bboxes_1[0],
                              x_flip=True, y_flip=True)

    pred_bboxes.append(pred_bboxes_1)
    pred_bboxes_.append(pred_bboxes)
    pred_labels_.append(pred_labels_1)
    pred_scores_.append(pred_scores_1)

    img_flip = imgflip(img, bbox, x_flip=False, y_flip=True)
    pred_bboxes_1, pred_labels_1, pred_scores_1, _ = faster_rcnn.predict(img_flip, [
        sizes], visualize=True)
    pred_bboxes_1 = bbox_flip(img_flip[0], pred_bboxes_1[0],
                              x_flip=True, y_flip=True)

    pred_bboxes2 = list()
    pred_bboxes2.append(pred_bboxes_1)
    pred_bboxes_.append(pred_bboxes2)
    pred_labels_.append(pred_labels_1)
    pred_scores_.append(pred_scores_1)

    img_flip = imgflip(img, bbox, x_flip=True, y_flip=False)
    pred_bboxes_1, pred_labels_1, pred_scores_1, _ = faster_rcnn.predict(img_flip, [
        sizes], visualize=True)
    pred_bboxes_1 = bbox_flip(img_flip[0], pred_bboxes_1[0],
                              x_flip=True, y_flip=True)
    pred_bboxes1 = list()
    pred_bboxes1.append(pred_bboxes_1)
    pred_bboxes_.append(pred_bboxes1)
    pred_labels_.append(pred_labels_1)
    pred_scores_.append(pred_scores_1)

    pred_bboxes_1, pred_labels_1, pred_scores_1, _ = faster_rcnn.predict(img, [
        sizes], visualize=True)

    pred_bboxes_.append(pred_bboxes_1)
    pred_labels_.append(pred_labels_1)
    pred_scores_.append(pred_scores_1)

    for i in range(4):
        for j in range(i):
            if len(pred_bboxes_[i][0])==0 or len(pred_bboxes_[j][0])==0:
                total_different += 1
                continue
            maps = diversity_map(pred_bboxes_[i], pred_labels_[i], pred_scores_[
                i], pred_bboxes_[j], pred_labels_[j], pred_scores_[j])

            if maps < thread:
                total_different += 1
    return total_different


def diversity_map(pred_bboxes, pred_labels, pred_scores, gt_bboxes1, gt_labels1, gt_difficult1):
    pred_bboxes_, pred_labels_, pred_scores_ = list(), list(), list()
    gt_bboxes_, gt_labels_, gt_difficults_ = list(), list(), list()
    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes1, gt_labels1,
        use_07_metric=True)
    return result['map']
