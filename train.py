#!/usr/bin/env python
# -*- coding:utf8 -*-
from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os
import numpy as np
import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize, Transform
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from PIL import Image
from matplotlib import pyplot as plt
from data.util import read_image
from example_mining import *
# import numpy as np
import torch as t
import cv2
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
import os
from utils.draw import draw
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# print(rlimit)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')
VOC_BBOX_LABEL_NAMES = opt.VOC_BBOX_LABEL_NAMES


def eval(dataloader, faster_rcnn, test_num=100):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_, id_) in enumerate(dataloader):
        if len(gt_bboxes_) == 0:
            continue
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_, _feature = faster_rcnn.predict(imgs, [
            sizes])

        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    # print len(pred_bboxes)
    # 这个评价函数是返回ap 和map值 其中传入的pred_bboxes格式为3维的数组的list格式，
    # 也就是说每个list都是一个3维数组(有batch的考量)
    # 其他的同理

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    if opt.is_distilltion == False:
        iteration_number = 10
        path = opt.voc_data_dir + '/ImageSets/Main/trainval.txt'
        datatxt = 0
        f = open(path, "r")
        for i in range(5000):
            if i % 500 == 0:
                datatxt = datatxt + 1
                f2 = open(opt.voc_data_dir + '/ImageSets/Main/' +
                          str(datatxt) + '.txt', "w")
            f2.write(f.readline())
    else:
        iteration_number = 1

    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    for jj in range(iteration_number):
        t.cuda.empty_cache()
        if jj > 0:
            opt.datatxt = str(int(opt.datatxt) + 1)
            opt.load_path = best_path
        # 样本挖掘
        print(opt.datatxt)
        if opt.is_example_mining == True and opt.load_path != None:
            if opt.example_type == 'mAP':
                example_mining_map(trainer, opt.datatxt)
            elif opt.example_type == 'loss':
                example_mining_loss(opt.datatxt)
            elif opt.example_type == 'diversity':
                example_mining_diversity(trainer, opt.datatxt)
            elif opt.example_type == 'mAP_diversity':
                example_mining_map_diversity(trainer, opt.datatxt)
            else:
                example_mining_map_loss(trainer, opt.datatxt)
            print('example mining completed')

        print('load data')
        dataset = Dataset(opt)
        dataloader = data_.DataLoader(dataset,
                                      batch_size=1,
                                      shuffle=True,
                                      # pin_memory=True,
                                      num_workers=opt.num_workers)
        testset = TestDataset(opt)
        test_dataloader = data_.DataLoader(testset,
                                           batch_size=1,
                                           num_workers=opt.test_num_workers,
                                           shuffle=False,
                                           pin_memory=True
                                           )

        testset_all = TestDataset(opt, 'test')
        test_all_dataloader = data_.DataLoader(testset_all,
                                               batch_size=1,
                                               num_workers=opt.test_num_workers,
                                               shuffle=False,
                                               pin_memory=True
                                               )

        # visdom 显示所有类别标签名
        trainer.vis.text(dataset.db.label_names, win='labels')
        best_map = 0

        lr_ = opt.lr
        # print(lr_)

        t.cuda.empty_cache()
        for epoch in range(opt.epoch):
            t.cuda.empty_cache()
            print('epoch=%d' % epoch)
            if opt.example_type != 'mAP':
                # 计算loss的数组初始化
                loss = np.zeros(10000)
                ID = list()

            # 重置混淆矩阵
            trainer.reset_meters()

            # tqdm可以在长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)，
            # 是一个快速、扩展性强
            for ii, (img, sizes, bbox_, label_, scale, id_) in enumerate(dataloader):
                if len(bbox_) == 0:
                    continue
                t.cuda.empty_cache()

                scale = at.scalar(scale)
                img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
                # 训练的就这一步 下面的都是打印的信息
                # 转化成pytorch能够计算的格式，转tensor格式
                if opt.is_distilltion == True:
                    # inx = str(id_[0])
                    # inx = int(inx[-5:])
                    # teacher_pred_bboxes = pred_bboxes[int(index[inx])]
                    # teacher_pred_labels = pred_labels[int(index[inx])]
                    # teacher_pred_features_ = pred_features[int(index[inx])]
                    teacher_pred_labels = np.load(
                        'label/' + str(id_[0]) + '.npy')
                    teacher_pred_bboxes = np.load(
                        'bbox/' + str(id_[0]) + '.npy')
                    teacher_pred_features_ = np.load(
                        'feature/' + str(id_[0]) + '.npy')
                    teacher_pred_bboxes = teacher_pred_bboxes.astype(
                        np.float32)
                    teacher_pred_labels = teacher_pred_labels.astype(np.int32)
                    teacher_pred_bboxes_ = at.totensor(teacher_pred_bboxes)
                    teacher_pred_labels_ = at.totensor(teacher_pred_labels)
                    teacher_pred_bboxes_ = teacher_pred_bboxes_.cuda()
                    teacher_pred_labels_ = teacher_pred_labels_.cuda()
                    teacher_pred_features_ = teacher_pred_features_.cuda()
                    losses = trainer.train_step(img, bbox, label, scale, epoch,
                                                teacher_pred_bboxes_, teacher_pred_labels_, teacher_pred_features_)
                else:
                    losses = trainer.train_step(img, bbox, label, scale, epoch)

                # 保存每一个样本的损失
                if opt.example_type != 'mAP':
                    ID += list(id_)
                    loss[ii] = losses.total_loss

                # visdom显示的信息
                if (ii + 1) % opt.plot_every == 0:
                    if os.path.exists(opt.debug_file):
                        ipdb.set_trace()

                    # plot loss
                    trainer.vis.plot_many(trainer.get_meter_data())

                    # plot groud truth bboxes
                    ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                    gt_img = visdom_bbox(ori_img_,
                                         at.tonumpy(bbox_[0]),
                                         at.tonumpy(label_[0]))
                    trainer.vis.img('gt_img', gt_img)
                    # plot predicti bboxes
                    _bboxes, _labels, _scores, _ = trainer.faster_rcnn.predict(
                        [ori_img_], visualize=True)
                    print(at.tonumpy(_bboxes[0]).reshape(-1).shape)
                    print(at.tonumpy(_labels[0]).shape)
                    pred_img = visdom_bbox(ori_img_,
                                           at.tonumpy(_bboxes[0]),
                                           at.tonumpy(_labels[0]).reshape(-1),
                                           at.tonumpy(_scores[0]))
                    trainer.vis.img('pred_img', pred_img)

                    # 混淆矩阵
                    # rpn confusion matrix(meter)
                    trainer.vis.text(
                        str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                    # roi confusion matrix
                    trainer.vis.text(
                        str(trainer.roi_cm.value().tolist()), win='roi_cm')
                    # trainer.vis.img('roi_cm', at.totensor(
                    # trainer.roi_cm.value(), False).float())

            eval_result = eval(test_dataloader, faster_rcnn,
                               test_num=opt.test_num)
            trainer.vis.plot('test_map', eval_result['map'])
            lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
            log_info = 'lr:{},ap:{}, map:{},loss:{}'.format(str(lr_),
                                                            str(eval_result['ap']),
                                                            str(eval_result['map']),
                                                            str(trainer.get_meter_data()))
            trainer.vis.log(log_info)

            # 保存最好结果并记住路径
            if eval_result['map'] > best_map:
                best_map = eval_result['map']
                best_path = trainer.save(best_map=best_map)
                if opt.example_type != 'mAP':
                    order = loss.argsort()[::-1]
                    f = open('loss.txt', "w")
                    for i in range(len(ID)):
                        f.write(ID[order[i]] + ' ' +
                                str(loss[order[i]]) + '\n')
                    f.close()

            if epoch == 20:
                #draw(test_dataloader, faster_rcnn, test_num=opt.test_num)
                save_name = trainer.save(best_map='20')
                f = open('result.txt', "a")
                result = eval(test_all_dataloader,
                              trainer.faster_rcnn, test_num=5000)
                f.write(opt.datatxt + '\n')
                f.write(save_name + '\n')
                f.write(result + '\n')
                f.close
                print(result)
                trainer.faster_rcnn.scale_lr(10)
                lr_ = lr_ * 10
                break

            # 每10轮加载前面最好权重，并且减少学习率
            if epoch % 20 == 15:
                trainer.save(best_map='15')
                trainer.load(best_path)
                trainer.faster_rcnn.scale_lr(opt.lr_decay)
                lr_ = lr_ * opt.lr_decay


if __name__ == '__main__':
    import fire

    fire.Fire()
