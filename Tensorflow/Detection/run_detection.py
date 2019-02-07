from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer
import Tensorflow.Util.DBUtil as dbUtil
import Tensorflow.Util.DictUtil as dictUtil

CLASSES = ('__background__','aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}


def vis_detections(im, class_name, dets, image_id, image_name, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    '''
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    '''
    type = dictUtil.get_classify_dict(class_name)
    list = []
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        point_x = bbox[0]
        point_y = bbox[1]
        width = bbox[2] - bbox[0]
        heigth = bbox[3] - bbox[1]

        sql = "INSERT INTO dl_detection_data(`id`,`name`,`top`,`left`,`width`,`height`,`type`,`score`) VALUES(%d,'%s',%f,%f,%f,%f,'%s',%f)" \
              % (image_id, image_name, point_y, point_x, width, heigth, type, score)
        list.append(sql)
        print(image_id, point_y, point_x, width, heigth, image_name, type, score)

    dbHelper = dbUtil.DBUtil()
    dbHelper.runSql(list)


def demo(sess, net, image_id, image_name):
    im_file = os.path.join(cfg.FLAGS2["data_dir"], 'image', image_name)
    im = cv2.imread(im_file)

    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    # print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, image_id, image_name, thresh=CONF_THRESH)


def parse_args():

    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset

    tfmodel = './model/faster_rcnn_model.ckpt'

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16(batch_size=1)
    # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(sess, "TEST", 21,
                            tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    images = [
        {'id': 1, 'name': '1.jpg'},
        {'id': 2, 'name': '2.jpg'}
    ]

    sqls = []
    for img in images:
        image_id = img["id"]
        sqls.append("DELETE FROM `dl_detection_data` WHERE `id` = '%s'" % image_id)
    dbHelper = dbUtil.DBUtil()
    dbHelper.runSql(sqls)

    for img in images:
        image_id = img["id"]
        image_name = img["name"]
        demo(sess, net, image_id, image_name)

    plt.show()
