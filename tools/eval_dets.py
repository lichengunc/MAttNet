from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import numpy as np
import h5py
import time
from pprint import pprint
import argparse

# model
import _init_paths
from layers.joint_match import JointMatching
from loaders.matt_dataset import MAttDetectionDataset
import models.eval_dets_utils as eval_utils

# torch
import torch
import torch.nn as nn

def load_model(checkpoint_path, opt):
    tic = time.time()
    model = JointMatching(opt)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    print('model loaded in %.2f seconds' % (time.time()-tic))
    return model

def evaluate(params):

    # load mode info
    model_prefix = osp.join('output', params['dataset_splitBy'], params['id'])
    infos = json.load(open(f'{model_prefix}/report.json'))
    model_opt = infos['opt']
    model_path = f'{model_prefix}/model.pth'
    model = load_model(model_path, model_opt)

    # set up loader
    data_json = osp.join('cache/prepro', params['dataset_splitBy'], 'data.json')
    dets_json = osp.join('data/detections', params['dataset_splitBy'], 
                         'res101_coco_minus_refer_notime_dets.json')
    iid_to_det_ids_json = osp.join('cache/prepro', params['dataset_splitBy'], 
                                   'iid_to_det_ids.json')
    eval_dataset = MAttDetectionDataset(
                    data_json, dets_json, iid_to_det_ids_json, 
                    params['split'], 
                    params['det_feats_dir'],
                    model_opt['seq_per_ref'], 
                    model_opt['visual_sample_ratio'],
                    model_opt['num_cxt'], model_opt['with_st'])

    # check model_info and params
    assert model_opt['dataset'] == params['dataset']
    assert model_opt['splitBy'] == params['splitBy']

    # evaluate on the split, 
    # predictions = [{sent_id, sent, gd_ann_id, pred_ann_id, pred_score, sub_attn, loc_attn, weights}]
    split = params['split']
    acc, predictions = eval_utils.eval_split(eval_dataset, model, model_opt)
    print('Comprehension on %s\'s %s (%s sents) is %.2f%%' % \
            (params['dataset_splitBy'], params['split'], len(predictions), acc*100.)) 

    # save
    out_dir = osp.join('cache', 'results', params['dataset_splitBy'], 'dets')
    if not osp.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = osp.join(out_dir, params['id']+'_'+params['split']+'.json')
    with open(out_file, 'w') as of:
        json.dump({'predictions': predictions, 'acc': acc}, of)

    # write to results.txt
    f = open('experiments/det_results.txt', 'a')
    f.write('[%s][%s], id[%s]\'s acc is %.2f%%\n' % \
            (params['dataset_splitBy'], params['split'], params['id'], acc*100.0))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
    parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
    parser.add_argument('--split', type=str, default='testA', help='split: testAB or val, etc')
    parser.add_argument('--id', type=str, default='0', help='model id name')
    parser.add_argument('--det_feats_dir', default='cache/feats/visual_grounding_det_coco')
    args = parser.parse_args()
    params = vars(args)

    # make other options
    params['dataset_splitBy'] = params['dataset'] + '_' + params['splitBy']
    evaluate(params)






