"""
We load Linjie's features from: cache/feats/visual_grounding_det_coco
Each feature is named as: visual_grounding_coco_000000581857.npz
containing {norm_bb, features, conf, soft_labels}
The order of extracted bbox and features should align with det_ids for each 
img_id.
We save this order for the use of REFER dataloader.
"""
import time
import pickle
import numpy as np
from pprint import pprint
from tqdm import tqdm
import json
import os.path as osp
import argparse


def recover_det_ids(denorm_bb, raw_bb, raw_det_ids):
    """
    Inputs:
    - denorm_bb  : [xywh], extracted from BUTD detectors.
    - raw_bb     : [xywh]
    - raw_det_ids
    Return:
    - ordered_det_ids: ordered by denorm_bb
    """
    assert denorm_bb.shape[0] == raw_bb.shape[0] 
    num_bb = denorm_bb.shape[0]
    ordered_det_ids = []
    for i in range(num_bb):
        ref_bb = denorm_bb[i]
        min_err, ix = 1e5, None
        for j in range(num_bb):
            if np.sum(np.abs(ref_bb - raw_bb[j])) < min_err:
                min_err, ix = np.sum(np.abs(ref_bb-raw_bb[j])), j
        ordered_det_ids.append(raw_det_ids[ix])
    return ordered_det_ids

def main(args):

    # dataset_splitBy 
    dataset = args.dataset
    splitBy = args.splitBy
    dataset_splitBy = args.dataset + '_' + args.splitBy
    print(f'Checking {dataset}')

    # Load imgs
    instances = json.load(open(osp.join(args.refer_dir, dataset, 
                        'instances.json')))
    # Load dets = [{det_id, image_id, category_id, category_name, box}]
    dets = json.load(open(osp.join(args.detections_dir, dataset_splitBy, 
                    'res101_coco_minus_refer_notime_dets.json')))
    
    # construct mapping
    Dets, iid_to_raw_det_ids = {}, {}
    for det in dets:
        Dets[det['det_id']] = det
        iid_to_raw_det_ids[det['image_id']] = iid_to_raw_det_ids.get(
            det['image_id'], []) + [det['det_id']]
    Imgs = {}
    for img in instances['images']:
        Imgs[img['id']] = img
   
    # Make iid_to_det_ids
    iid_to_det_ids = {}
    warning_img_ids = set()
    img_ids = list(Imgs.keys())
    for img_id in tqdm(img_ids):
        raw_det_ids = iid_to_raw_det_ids[img_id]
        # raw_det_bb 
        raw_det_bb = np.array([Dets[det_id]['box'] for det_id in raw_det_ids])
        # denorm_bb
        im_width = Imgs[img_id]['width']
        im_height = Imgs[img_id]['height']
        img_feat = np.load(osp.join(args.feats_dir, 
                        f'visual_grounding_det_coco_{int(img_id):012}.npz'))
        norm_bb = img_feat['norm_bb']
        x1, x2 = norm_bb[:, 0] * im_width, norm_bb[:, 2] * im_width
        y1, y2 = norm_bb[:, 1] * im_height, norm_bb[:, 3] * im_height
        w, h = norm_bb[:, 4] * im_width, norm_bb[:, 5] * im_height
        denorm_bb = np.stack([x1, y1, w, h], axis=1)  # (n,4)
        # re-order det_ids
        ordered_det_ids = recover_det_ids(denorm_bb, raw_det_bb, raw_det_ids)
        # check difference
        if set(ordered_det_ids) != set(raw_det_ids):
            print('Please check img_id[%s]'%img_id)
            warning_img_ids.add(img_id)
        # check length of det_ids
        assert len(ordered_det_ids) == len(raw_det_ids)
        # add to iid_to_det_ids
        iid_to_det_ids[img_id] = ordered_det_ids

    print('%s images contain dupicated bounding boxes.' % len(warning_img_ids))
    pprint(list(warning_img_ids))

    # save    
    output_file = osp.join(args.cache_dir, 'prepro', dataset_splitBy, 
                           'iid_to_det_ids.json')
    with open(output_file, 'w') as f:
        json.dump({'iid_to_det_ids': iid_to_det_ids}, f)
    print('%s iid_to_det_ids saved in %s.' % (len(iid_to_det_ids), output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refer_dir', 
                    default='data', 
                    help='folder saving all downloaded refer datasets')
    parser.add_argument('--detections_dir',
                    default='data/detections',
                    help='folder saving detection results')
    parser.add_argument('--dataset',
                    default='refcoco',
                    help='dataset name')
    parser.add_argument('--splitBy',
                    default='unc',
                    help='institute proving the split')
    parser.add_argument('--feats_dir', 
                    default='cache/feats/visual_grounding_det_coco',
                    help='folder saving butd features.')
    parser.add_argument('--cache_dir', 
                    default='cache',
                    help='output folder saving img_id --> [ann_id]')
    args = parser.parse_args()
    main(args)