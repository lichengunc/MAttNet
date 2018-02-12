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
this_dir = osp.dirname(__file__)

# mrcn path
import _init_paths
from mrcn import inference

# dataloader
sys.path.insert(0, osp.join(this_dir, '../lib/loaders'))
from loader import Loader


def main(args):
  dataset_splitBy = args.dataset + '_' + args.splitBy
  if not osp.isdir(osp.join('cache/feats/', dataset_splitBy)):
    os.makedirs(osp.join('cache/feats/', dataset_splitBy))

  # Image Directory
  if 'coco' in dataset_splitBy:
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

  # load mrcn model
  mrcn = inference.Inference(args)
  imdb = mrcn.imdb

  # feats_h5
  feats_dir = osp.join('cache/feats', dataset_splitBy, 'mrcn', '%s_%s_%s' % (args.net_name, args.imdb_name, args.tag))
  if not osp.isdir(feats_dir):
    os.makedirs(feats_dir)

  # extract
  for i, image in enumerate(images):
    file_name = image['file_name']
    img_path = osp.join(IMAGE_DIR, file_name)
    feat, im_info = mrcn.extract_head(img_path)
    feat = feat.data.cpu().numpy()

    # write
    feat_h5 = osp.join(feats_dir, str(image['image_id'])+'.h5')
    f = h5py.File(feat_h5, 'w')
    f.create_dataset('head', dtype=np.float32, data=feat)
    f.create_dataset('im_info', dtype=np.float32, data=im_info)
    f.close()
    if i % 10 == 0:
      print('%s/%s image_id[%s] size[%s] im_scale[%.2f] writen.' % (i+1, len(images), image['image_id'], feat.shape, im_info[0][2]))

  print('Done.')


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


