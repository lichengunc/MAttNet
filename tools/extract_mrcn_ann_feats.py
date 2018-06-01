from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path as osp
import sys
import json
import time
import numpy as np
import h5py
import pprint
from scipy.misc import imread, imresize
import cv2

import torch
from torch.autograd import Variable

# mrcn path
import _init_paths
from mrcn import inference_no_imdb

# dataloader
from loaders.loader import Loader


# box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))

def image_to_head(head_feats_dir, image_id):
  """Returns
  head: float32 (1, 1024, H, W)
  im_info: float32 [[im_h, im_w, im_scale]]
  """
  feats_h5 = osp.join(head_feats_dir, str(image_id)+'.h5')
  feats = h5py.File(feats_h5, 'r')
  head, im_info = feats['head'], feats['im_info']
  return np.array(head), np.array(im_info)

def ann_to_fc7(mrcn, ann, net_conv, im_info):
  """
  Arguments:
    ann: object instance
    net_conv: float32 (1, 1024, H, W)
    im_info: float32 [[im_h, im_w, im_scale]]
  Returns:
    fc7: Variable(cuda) (1, 2048)
  """
  box = np.array([ann['box']])  # [[xywh]]
  box = xywh_to_xyxy(box)  # [[x1y1x2y2]]
  fc7 = mrcn.box_to_fc7(Variable(torch.from_numpy(net_conv).cuda()), im_info, box)  # (1, 2048)
  return fc7

def ann_to_pool5_fc7(mrcn, ann, net_conv, im_info):
  """
  Arguments:
    ann: object instance
    net_conv: float32 (1, 1024, H, W)
    im_info: float32 [[im_h, im_w, im_scale]]
  Returns:
    pool5: Variable(cuda) (1, 1024)
    fc7  : Variable(cuda) (1, 2048)
  """
  box = np.array([ann['box']])  # [[xywh]]
  box = xywh_to_xyxy(box)  # [[x1y1x2y2]]
  pool5, fc7 = mrcn.box_to_pool5_fc7(Variable(torch.from_numpy(net_conv).cuda()), im_info, box)  # (1, 2048)
  return pool5, fc7

def main(args):
  dataset_splitBy = args.dataset + '_' + args.splitBy
  if not osp.isdir(osp.join('cache/feats/', dataset_splitBy)):
    os.makedirs(osp.join('cache/feats/', dataset_splitBy))

  # Image Directory
  if 'coco' or 'combined' in dataset_splitBy:
    IMAGE_DIR = 'data/images/mscoco/images/train2014'
  elif 'clef' in dataset_splitBy:
    IMAGE_DIR = 'data/images/saiapr_tc-12'
  else:
    print('No image directory prepared for ', args.dataset)
    sys.exit(0)

  # load dataset
  data_json = osp.join('cache/prepro', dataset_splitBy, 'data.json')
  data_h5 = osp.join('cache/prepro', dataset_splitBy, 'data.h5')
  loader = Loader(data_json, data_h5)
  images = loader.images
  anns = loader.anns
  num_anns = len(anns)
  assert sum([len(image['ann_ids']) for image in images]) == num_anns

  # load mrcn model
  mrcn = inference_no_imdb.Inference(args)

  # feats_h5
  # feats_h5 = osp.join('cache/feats', dataset_splitBy, args.file_name)
  file_name = '%s_%s_%s_ann_feats.h5' % (args.net_name, args.imdb_name, args.tag)
  feats_h5 = osp.join('cache/feats', dataset_splitBy, 'mrcn', file_name)

  f = h5py.File(feats_h5, 'w')
  pool5_set = f.create_dataset('pool5', (num_anns, 1024), dtype=np.float32)
  fc7_set = f.create_dataset('fc7', (num_anns, 2048), dtype=np.float32)

  # extract
  feats_dir = '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag)
  head_feats_dir = osp.join('cache/feats/', dataset_splitBy, 'mrcn', feats_dir)
  for i, image in enumerate(images):
    image_id = image['image_id']
    net_conv, im_info = image_to_head(head_feats_dir, image_id)
    ann_ids = image['ann_ids']
    for ann_id in ann_ids:
      ann = loader.Anns[ann_id]
      ann_pool5, ann_fc7 = ann_to_pool5_fc7(mrcn, ann, net_conv, im_info)
      ann_h5_id = ann['h5_id']
      pool5_set[ann_h5_id] = ann_pool5.data.cpu().numpy()
      fc7_set[ann_h5_id] = ann_fc7.data.cpu().numpy()
    if i % 20 == 0:
      print('%s/%s done.' % (i+1, len(images)))

  f.close()
  print('%s written.' % feats_h5)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
  parser.add_argument('--net_name', default='res101')
  parser.add_argument('--iters', default=1250000, type=int)
  parser.add_argument('--tag', default='notime')

  parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
  parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
  args = parser.parse_args()
  main(args)


