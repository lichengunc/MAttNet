"""
args: net, iters, tag
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import time
import numpy as np
import pprint
from scipy.misc import imread, imresize
import cv2

import torch
from torch.autograd import Variable

# mrcn imports
import _init_paths
from datasets.factory import get_imdb
from model.config import cfg, cfg_from_file, cfg_from_list
from model.bbox_transform import clip_boxes, bbox_transform_inv
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from utils.blob import im_list_to_blob
from utils.mask_utils import recover_masks
from pycocotools import mask as COCOmask

# mrcn dir
this_dir = osp.dirname(__file__)
mrcn_dir = osp.join(this_dir, '..', '..', 'pyutils', 'mask-faster-rcnn')

def get_imdb_name(imdb_name):
  if imdb_name in ['refcoco', 'refcocog']:
    return {'TRAIN_IMDB': '%s_train+%s_val' % (imdb_name, imdb_name),
            'TEST_IMDB' : '%s_test' % imdb_name}
  elif imdb_name == 'coco_minus_refer':
    return {'TRAIN_IMDB': "coco_2014_train_minus_refer_valtest+coco_2014_valminusminival",
            'TEST_IMDB' : "coco_2014_minival"}

class Inference:

  def __init__(self, args):

    self.imdb_name = args.imdb_name
    self.net_name = args.net_name
    self.tag = args.tag
    self.iters = args.iters

    # Config
    cfg_file = osp.join(mrcn_dir, 'experiments/cfgs/%s.yml' % self.net_name)
    cfg_list = ['ANCHOR_SCALES', [4,8,16,32], 'ANCHOR_RATIOS', [0.5,1,2]]
    if cfg_file is not None: cfg_from_file(cfg_file)
    if cfg_list is not None: cfg_from_list(cfg_list)
    print('Using config:')
    pprint.pprint(cfg)

    # Load network
    self.num_classes = 81  # hard code this
    self.net = self.load_net()

  def load_net(self):
    # Load network
    if self.net_name == 'vgg16':
      net = vgg16(batch_size=1)
    elif self.net_name == 'res101':
      net = resnetv1(batch_size=1, num_layers=101)
    else:
      raise NotImplementedError

    net.create_architecture(self.num_classes, tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)
    net.eval()
    net.cuda()

    # Load model
    model = osp.join(mrcn_dir, 'output/%s/%s/%s/%s_mask_rcnn_iter_%s.pth' % \
      (self.net_name, get_imdb_name(self.imdb_name)['TRAIN_IMDB'], self.tag, self.net_name, self.iters))
    assert osp.isfile(model), model
    net.load_state_dict(torch.load(model))
    print('pretrained-model loaded from [%s].' % model)

    return net

  def predict(self, img_path):
    # return scores/probs (num_rois, 81), pred_boxes (num_rois, 81*4)
    # in numpy
    im = cv2.imread(img_path)
    blobs, im_scales = self._get_blobs(im) 
    im_blob = blobs['data']  # (1, iH, iW, 3)
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    
    # test_image returns cls_score, cls_prob, bbox_pred, rois, net_conv
    _, scores, bbox_pred, rois, _ = self.net.test_image(blobs['data'], blobs['im_info'])

    boxes = rois[:, 1:5] / im_scales[0]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred
      pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
      pred_boxes = self._clip_boxes(pred_boxes, im.shape)
    else:
      # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


  def boxes_to_masks(self, img_path, boxes, labels):
    """
    Arguments:
    - img_path: img_file
    - boxes   : ndaray [[xyxy]] (n, 4) in original image
    - labels  : ndarray (n, )
    Return:
    - masks   : (n, ih, iw) uint8 [0,1]
    - rles    : list of rle instance
    """ 
    im = cv2.imread(img_path)
    blobs, im_scales = self._get_blobs(im)
    im_blob = blobs['data']  # (1, iH, iW, 3) 
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    # forward
    self.net.test_image(blobs['data'], blobs['im_info'])

    # net_conv
    net_conv = self.net._predictions['net_conv']

    # run
    mask_prob = self.net._predict_masks_from_boxes_and_labels(net_conv, boxes*im_scales[0], labels)
    mask_prob = mask_prob.data.cpu().numpy()
    masks = recover_masks(mask_prob, boxes, im.shape[0], im.shape[1])  # (N, ih, iw) uint8 [0-255]
    masks = (masks > 122.).astype(np.uint8)  # (N, ih, iw) uint8 [0,1]

    # encode to rles
    rles = []
    for m in masks:
      rle = COCOmask.encode(np.asfortranarray(m))
      rles += [rle]

    return masks, rles

  
  def extract_head(self, img_path):
    # extract head (1, 1024, im_height*scale/16.0, im_width*scale/16.0) in Variable cuda float
    # and im_info [[ih, iw, scale]] in float32 ndarray
    im = cv2.imread(img_path)
    blobs, im_scales = self._get_blobs(im) 
    head_feat = self.net.extract_head(blobs['data'])
    im_info = np.array([[blobs['data'].shape[1], blobs['data'].shape[2], im_scales[0]]])
    return head_feat, im_info.astype(np.float32)

  def head_to_prediction(self, net_conv, im_info):
    """
    Arguments:
      net_conv (Variable): (1, 1024, H, W)
      im_info (float) : [[ih, iw, scale]]
    Returns:
      scores (ndarray): (num_rois, 81)
      pred_boxes (ndarray): (num_rois, 81*4) in original image size
    """
    self.net.eval()
    self.net._mode = 'TEST'

    # predict rois, cls_prob and bbox_pred
    self.net._im_info = im_info
    self.net._anchor_component(net_conv.size(2), net_conv.size(3))
    rois = self.net._region_proposal(net_conv)
    if cfg.POOLING_MODE == 'crop':
      pool5 = self.net._crop_pool_layer(net_conv, rois)
    else:
      pool5 = self.net._roi_pool_layer(net_conv, rois)
    fc7 = self.net._head_to_tail(pool5)
    cls_prob, bbox_pred = self.net._region_classification(fc7)
    
    # add mean and std to bbox_pred if any
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
      stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self.num_classes).unsqueeze(0).expand_as(bbox_pred)
      means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self.num_classes).unsqueeze(0).expand_as(bbox_pred)
      bbox_pred = bbox_pred.mul(Variable(stds)).add(Variable(means))

    # convert to numpy
    scores = cls_prob.data.cpu().numpy()
    rois = rois.data.cpu().numpy()
    bbox_pred = bbox_pred.data.cpu().numpy()

    # regress boxes
    boxes = rois[:, 1:5] / im_info[0][2]
    if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred
      pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
      pred_boxes = self._clip_boxes(pred_boxes, im_info[0][:2])
    else:
      # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes

  def box_to_spatial_fc7(self, net_conv, im_info, ori_boxes):
    """
    Arguments:
      net_conv (Variable)  : (1, 1024, H, W)
      im_info (float32)    : [[ih, iw, scale]]
      ori_boxes (float32)  : (n, 4) [x1y1x2y2]
    Returns:
      pool5 (float)        : (n, 1024, 7, 7)
      spatial_fc7 (float)  : (n, 2048, 7, 7)
    """
    self.net.eval()
    self.net._mode = 'TEST'

    # make rois
    batch_inds = Variable(net_conv.data.new(ori_boxes.shape[0], 1).zero_())
    scaled_boxes = (ori_boxes * im_info[0][2]).astype(np.float32)
    scaled_boxes = Variable(torch.from_numpy(scaled_boxes).cuda())
    rois = torch.cat([batch_inds, scaled_boxes], 1)

    # pool fc7
    if cfg.POOLING_MODE == 'crop':
      pool5 = self.net._crop_pool_layer(net_conv, rois)
    else:
      pool5 = self.net._roi_pool_layer(net_conv, rois)  # (n, 1024, 7, 7)

    spatial_fc7 = self.net.resnet.layer4(pool5)  # (n, 2048, 7, 7), equavalent to _head_to_tail
    return pool5, spatial_fc7

  def box_to_pool5_fc7(self, net_conv, im_info, ori_boxes):
    """
    Arguments:
      net_conv (Variable)  : (1, 1024, H, W)
      im_info (float32)    : [[ih, iw, scale]]
      ori_boxes (float32)  : (n, 4) [x1y1x2y2]
    Returns:
      pool5 (float): (n, 1024)
      fc7 (float)  : (n, 2048)
    """
    self.net.eval()
    self.net._mode = 'TEST'

    # make rois
    batch_inds = Variable(net_conv.data.new(ori_boxes.shape[0], 1).zero_())
    scaled_boxes = (ori_boxes * im_info[0][2]).astype(np.float32)
    scaled_boxes = Variable(torch.from_numpy(scaled_boxes).cuda())
    rois = torch.cat([batch_inds, scaled_boxes], 1)

    # pool fc7
    if cfg.POOLING_MODE == 'crop':
      pool5 = self.net._crop_pool_layer(net_conv, rois)
    else:
      pool5 = self.net._roi_pool_layer(net_conv, rois)

    fc7 = self.net._head_to_tail(pool5)  # (n, 2048, 7, 7)
    pool5 = pool5.mean(3).mean(2)
    fc7 = fc7.mean(3).mean(2)  # (n, 2048)
    return pool5, fc7

  def box_to_fc7(self, net_conv, im_info, ori_boxes):
    """
    Arguments:
      net_conv (Variable)  : (1, 1024, H, W)
      im_info (float32)    : [[ih, iw, scale]]
      ori_boxes (float32)  : (n, 4) [x1y1x2y2]
    Returns:
      fc7 (float) : (n, 2048)
    """
    self.net.eval()
    self.net._mode = 'TEST'

    # make rois
    batch_inds = Variable(net_conv.data.new(ori_boxes.shape[0], 1).zero_())
    scaled_boxes = (ori_boxes * im_info[0][2]).astype(np.float32)
    scaled_boxes = Variable(torch.from_numpy(scaled_boxes).cuda())
    rois = torch.cat([batch_inds, scaled_boxes], 1)

    # pool fc7
    if cfg.POOLING_MODE == 'crop':
      pool5 = self.net._crop_pool_layer(net_conv, rois)
    else:
      pool5 = self.net._roi_pool_layer(net_conv, rois)

    fc7 = self.net._head_to_tail(pool5) # (n, 2048, 7, 7)
    fc7 = fc7.mean(3).mean(2)  # (n, 2048)
    return fc7

  def spatial_fc7_to_prediction(self, spatial_fc7, im_info, ori_boxes):
    """Only used for testing. Testing the above box_to_fc7 [passed]"""
    cls_prob, bbox_pred = self.net._region_classification(spatial_fc7)

    # make rois
    batch_inds = Variable(spatial_fc7.data.new(ori_boxes.shape[0], 1).zero_())
    scaled_boxes = (ori_boxes * im_info[0][2]).astype(np.float32)
    scaled_boxes = Variable(torch.from_numpy(scaled_boxes).cuda())
    rois = torch.cat([batch_inds, scaled_boxes], 1)
    
    # add mean and std to bbox_pred if any
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
      stds = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_STDS).repeat(self.num_classes).unsqueeze(0).expand_as(bbox_pred)
      means = bbox_pred.data.new(cfg.TRAIN.BBOX_NORMALIZE_MEANS).repeat(self.num_classes).unsqueeze(0).expand_as(bbox_pred)
      bbox_pred = bbox_pred.mul(Variable(stds)).add(Variable(means))

    # convert to numpy
    scores = cls_prob.data.cpu().numpy()
    rois = rois.data.cpu().numpy()
    bbox_pred = bbox_pred.data.cpu().numpy()

    # regress boxes
    boxes = rois[:, 1:5] / im_info[0][2]
    if cfg.TEST.BBOX_REG:
      # Apply bounding-box regression deltas
      box_deltas = bbox_pred
      pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
      pred_boxes = self._clip_boxes(pred_boxes, im_info[0][:2])
    else:
      # Simply repeat the boxes, once for each class
      pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes

  def _get_image_blob(self, im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
      im_scale = float(target_size) / float(im_size_min)
      # Prevent the biggest axis from being more than MAX_SIZE
      if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
      im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
              interpolation=cv2.INTER_LINEAR)
      im_scale_factors.append(im_scale)
      processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

  def _get_blobs(self, im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = self._get_image_blob(im)

    return blobs, im_scale_factors

  def _clip_boxes(self, boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes









