#!/usr/bin/env python
# -*- coding:utf-8 -*-
from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    min_size = 600  # image resize
    max_size = 1000  # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 100

    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'
    threshold = 0.5
    test_num = 500

    # ensemble
    ensemble_test_numebr = 5000
    ensemble_use_map = True
    ensemble_type='mAP'
    score_thresh = 0.05
    nms_type = 'soft_nms'  # soft_nms nms
    nms_use_label = True
    # model

    is_distilltion = False
    testtxt = 'test'
    datatxt = '2'
    load_path = 'checkpoints/fasterrcnn_12201310_20'
    # load_path = None

    # example mmining
    example_sort = 'max'
    mining_number = 100
    is_example_mining = False
    example_type = 'mAP'  # (diversity mAP loss mAP_diversity map_loss)

    caffe_pretrain = True  # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    # data
    voc_2007 = True
    if voc_2007 == True:
        voc_data_dir = '/media/chenli/E/VOCdevkit/VOC2007'
        VOC_BBOX_LABEL_NAMES = (
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'pottedplant',
            'sheep',
            'sofa',
            'train',
            'tvmonitor')
    else:
        voc_data_dir = '../../BCCD'
        VOC_BBOX_LABEL_NAMES = (
            'wbc',
            'rbc',
            'platelets'
        )

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}


opt = Config()
