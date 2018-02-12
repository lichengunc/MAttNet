"""
1) masks information: cache/detections/dataset_splitBy/res101_coco_minus_refer_notime_masks.json
    contains [{det_id, h5_id, box, category_id, category_name, image_id, rle, size, score}]
2) eval_dets results: cache/results/dataset_splitBy/dets/id_split.json, contains [predictions, acc], 
    where predictions = [{sent_id, sent, gd_box, pred_det_id, pred_box, pred_score, ...}]
We make det_to_rle then use refer to check overall mask IoU.
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

# add paths
import _init_paths
from refer import REFER
from pycocotools import mask as maskUtils

# compute IoU
def computeIoU(pred_rle, gd_rle):
  pred_seg = maskUtils.decode(pred_rle) # (H, W)
  gd_seg = maskUtils.decode(gd_rle)     # (H, W)
  I = np.sum(np.logical_and(pred_seg, gd_seg))
  U = np.sum(np.logical_or(pred_seg, gd_seg))
  return I, U

# get mask from ann
def annToRLE(ann, h, w):
  segm = ann['segmentation']
  if type(segm) == list:
    # polygon -- a single object might consist of multiple parts
    # we merge all parts into one mask rle code
    rles = maskUtils.frPyObjects(segm, h, w)
    rle = maskUtils.merge(rles)
  elif type(segm['counts']) == list:
    # uncompressed RLE
    rle = maskUtils.frPyObjects(segm, h, w)
  else:
    # rle
    rle = ann['segmentation']
  return rle


def main(args):
  params = vars(args)

  # load refer
  refer = REFER('data', args.dataset, args.splitBy)

  # load masks info
  dataset_splitBy = args.dataset + '_' + args.splitBy
  masks_info_path = osp.join('cache/detections', dataset_splitBy, '%s_%s_%s_masks.json' % \
                             (args.net_name, args.imdb_name, args.tag))
  masks_info = json.load(open(masks_info_path))

  # make det_to_rle
  det_to_rle = {}
  for mask in masks_info:
    det_id = mask['det_id']
    rle = mask['rle']
    det_to_rle[det_id] = rle

  # load comprehension results
  results_path = osp.join('cache/results', dataset_splitBy, 'dets', args.id+'_'+args.split+'.json')
  results = json.load(open(results_path, 'r'))

  cum_I, cum_U = 0, 0
  eval_seg_iou_list = [.5, .6, .7, .8, .9]
  seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
  seg_total = 0
  for pid, pred in enumerate(results['predictions']):
    sent_id = pred['sent_id']
    ref = refer.sentToRef[sent_id]
    ann = refer.Anns[ref['ann_id']]
    image = refer.Imgs[ref['image_id']]
    gd_rle = annToRLE(ann, image['height'], image['width'])
    pred_rle = det_to_rle[pred['pred_det_id']]

    # add to pred
    pred['rle'] = pred_rle
    
    # compute iou
    I, U = computeIoU(pred_rle, gd_rle)
    cum_I += I
    cum_U += U
    for n_eval_iou in range(len(eval_seg_iou_list)):
      eval_seg_iou = eval_seg_iou_list[n_eval_iou]
      seg_correct[n_eval_iou] += (I*1.0/U >= eval_seg_iou)
    seg_total += 1
    print('%s/%s expressions evaluated, iou=%.2f.' % (pid+1, len(results['predictions']), I*1.0/U))

  # print 
  print('Final results on [%s][%s]' % (dataset_splitBy, args.split))
  results_str = ''
  for n_eval_iou in range(len(eval_seg_iou_list)):
    results_str += '    precision@%s = %.2f\n' % \
      (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou]*100./seg_total)
  results_str += '    ovearall IoU = %.2f\n' % (cum_I*100./cum_U)
  print(results_str)

  # save results
  save_dir = osp.join('cache/results', dataset_splitBy, 'masks')
  if not osp.isdir(save_dir):
    os.makedirs(save_dir)

  results['iou'] = cum_I*1./cum_U
  assert 'rle' in results['predictions'][0]
  with open(osp.join(save_dir, args.id+'_'+args.split+'.json'), 'w') as f:
    json.dump(results, f)

  # write to results.txt
  f = open('experiments/mask_results.txt', 'a')
  f.write('[%s][%s], id[%s]\'s iou is:\n%s' % \
          (dataset_splitBy, params['split'], params['id'], results_str))


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--imdb_name', default='coco_minus_refer', help='image databased trained on.')
  parser.add_argument('--net_name', default='res101')
  parser.add_argument('--iters', default=1250000, type=int)
  parser.add_argument('--tag', default='notime')

  parser.add_argument('--id', default='mrcn_cmr_with_st', type=str, help='model id name')
  parser.add_argument('--dataset', default='refcoco', type=str, help='refcoco/refcoco+/refcocog')
  parser.add_argument('--splitBy', default='unc', type=str, help='unc/google')
  parser.add_argument('--split', default='val', type=str, help='val/testA/testB/test')

  args = parser.parse_args()
  main(args)

