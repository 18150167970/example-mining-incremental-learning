#!/usr/bin/env python
# -*- coding:utf8 -*-
from __future__ import absolute_import

import os
import torch as t
import cv2
import numpy as np

from utils import array_tool as at
from utils.config import opt
from tqdm import tqdm
import six


def draw(dataloader, faster_rcnn, test_num=100):
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_, _feature = faster_rcnn.predict(imgs, [
                                                                                 sizes])
        img_file = opt.voc_data_dir + '/JPEGImages/' + str(id_[0]) + '.jpg'
        image = cv2.imread(img_file)
        # 转成 numpy格式
        bboxs = at.tonumpy(pred_bboxes_[0])
        name = at.tonumpy(pred_labels_[0]).reshape(-1)
        score = at.tonumpy(pred_scores_[0]).reshape(-1)

        # 保存测试集每一轮预测的结果 最好加个epoch判断 每10轮保存一次 不然太浪费时间
        for i in range(len(name)):
            xmin = int(round(float(bboxs[i, 1])))
            ymin = int(round(float(bboxs[i, 0])))
            xmax = int(round(float(bboxs[i, 3])))
            ymax = int(round(float(bboxs[i, 2])))
            if score[i] <= opt.threshold:
                continue
            cv2.rectangle(image, (xmin, ymin),
                          (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(image, opt.VOC_BBOX_LABEL_NAMES[name[i]], (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
            cv2.putText(image, str(score[i])[0:3], (xmin + 30, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
        cv2.imwrite('result/' + str(id_[0]) + '.jpg', image)


def draw_predict(pred_bboxes_, pred_labels_, pred_scores_):
    pred_bboxes1 = iter(pred_bboxes_)
    pred_labels1 = iter(pred_labels_)
    pred_scores1 = iter(pred_scores_)
    if opt.nms_type == 'soft_nms':
        write_path = 'result/'
    else:
        write_path = 'result_nms/'
    if opt.nms_use_label == True:
        write_path = 'label_' + write_path
    print (write_path)
    f = open('/media/chenli/E/VOCdevkit/VOC2007/ImageSets/Main/test2.txt')
    for pred_bbox, pred_label, pred_score in six.moves.zip(pred_bboxes1, pred_labels1, pred_scores1):
        id_ = f.readline()[:-1]
        # print id_
        img_file = '/media/chenli/E/VOCdevkit/VOC2007/JPEGImages/' + \
            str(id_) + '.jpg'
        image = cv2.imread(img_file)
        # 转成 numpy格式
        bboxs = at.tonumpy(pred_bbox)
        name = at.tonumpy(pred_label).reshape(-1)
        score = at.tonumpy(pred_score).reshape(-1)
        # 保存测试集每一轮预测的结果 最好加个epoch判断 每10轮保存一次 不然太浪费时间
        for i in range(len(name)):
            xmin = int(round(float(bboxs[i, 1])))
            ymin = int(round(float(bboxs[i, 0])))
            xmax = int(round(float(bboxs[i, 3])))
            ymax = int(round(float(bboxs[i, 2])))
            if score[i] <= opt.threshold:
                continue
            cv2.rectangle(image, (xmin, ymin),
                          (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(image, opt.VOC_BBOX_LABEL_NAMES[name[i]], (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
            cv2.putText(image, str(score[i])[0:3], (xmin + 30, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)

        cv2.imwrite(write_path + str(id_) + '.jpg', image)
