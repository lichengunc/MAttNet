"""
Given detected results from:
cache/detections/refcoco_unc/{net}_{imdb}_{tag}_dets.json has
0. dets: list of {det_id, box, image_id, category_id, category_name, score}

We further run mask r-cnn on each box to fetch the segmentation, and output
0. dets: list of {det_id, box, image_id, category_id, category_name, score, rle}
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import numpy as np
import argparse
import json
import torch
from scipy.misc import imread

# add paths
import _init_paths
from model.nms_wrapper import nms
from mrcn import inference, inference_no_imdb


# box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def xyxy_to_xywh(boxes):
  """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def main(args):
  # Image Directory
  params = vars(args)
  dataset_splitBy = params['dataset'] + '_' + params['splitBy']
  if 'coco' or 'combined' in dataset_splitBy:
    IMAGE_DIR = 'data/images/mscoco/images/train2014'
  elif 'clef' in dataset_splitBy:
    IMAGE_DIR = 'data/images/saiapr_tc-12'
  else:
    print('No image directory prepared for ', args.dataset)
    sys.exit(0)

  # make save dir
  save_dir = osp.join('cache/detections', dataset_splitBy)
  if not osp.isdir(save_dir):
    os.makedirs(save_dir)
  print(save_dir)

  # get mrcn instance
  mrcn = inference.Inference(args)

  # load detections = [{det_id, box, image_id, category_id, category_name, score}]
  save_dir = osp.join('cache/detections', args.dataset+'_'+args.splitBy)
  detections = json.load(open(osp.join(save_dir, args.dets_file_name)))
  image_to_dets = {}
  for det in detections:
    image_id = det['image_id']
    if image_id not in image_to_dets:
      image_to_dets[image_id] = []
    image_to_dets[image_id] += [det]

  # run mask rcnn
  for i, image_id in enumerate(image_to_dets.keys()):
    dets = image_to_dets[image_id]

    img_path = osp.join(IMAGE_DIR, 'COCO_train2014_'+str(image_id).zfill(12)+'.jpg')
    assert osp.isfile(img_path), img_path

    boxes = np.array([det['box'] for det in dets], dtype=np.float32)
    boxes = xywh_to_xyxy(boxes)

    labels = [mrcn.imdb._class_to_ind[det['category_name']] for det in dets]
    labels = np.array(labels, dtype=np.int32)

    masks, rles = mrcn.boxes_to_masks(img_path, boxes, labels)

    # add rles to det
    for ix, det in enumerate(dets):
      det['rle'] = rles[ix]

    print('%s/%s done.' % (i, len(image_to_dets)))

  # save dets.json = [{det_id, box, image_id, score}]
  # to cache/detections/
  save_path = osp.join(save_dir, args.dets_file_name[:-10] + '_masks.json')
  with open(save_path, 'w') as f:
    json.dump(detections, f)


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
  parser.add_argument('--net_name', default='res101')
  parser.add_argument('--iters', default=1250000, type=int)
  parser.add_argument('--tag', default='notime')

  parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
  parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')
  parser.add_argument('--dets_file_name', default='res101_coco_minus_refer_notime_dets.json', type=str)

  args = parser.parse_args()
  main(args)

