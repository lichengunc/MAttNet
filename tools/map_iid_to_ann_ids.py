"""
We load Linjie's features from: cache/feats/visual_grounding_coco_gt
Each feature is named as: visual_grounding_coco_000000581857.npz
containing {norm_bb, features, conf, soft_labels}
The order of extracted bbox and features should align with ann_ids for each 
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


def recover_ann_ids(denorm_bb, raw_bb, raw_ann_ids):
    """
    Inputs:
    - denorm_bb  : [xywh], extracted from BUTD detectors.
    - raw_bb     : [xywh]
    - raw_ann_ids
    Return:
    - ordered_ann_ids: ordered by denorm_bb
    """
    assert denorm_bb.shape[0] == raw_bb.shape[0] 
    num_bb = denorm_bb.shape[0]
    ordered_ann_ids = []
    for i in range(num_bb):
        ref_bb = denorm_bb[i]
        min_err, ix = 1e5, None
        for j in range(num_bb):
            if np.sum(np.abs(ref_bb - raw_bb[j])) < min_err:
                min_err, ix = np.sum(np.abs(ref_bb-raw_bb[j])), j
        ordered_ann_ids.append(raw_ann_ids[ix])
    return ordered_ann_ids

def main(args):

    # Load all instances from refcoco, refcoco+ and refcocog
    tic = time.time()
    iid_to_ann_ids = {}
    warning_img_ids = set()
    for dataset in ['refcoco', 'refcoco+', 'refcocog']:
        print('Checking %s...' % dataset)
        instances = json.load(open(osp.join(args.refer_dir, dataset, 
                        'instances.json')))
        Anns, Imgs, iid_to_raw_ann_ids = {}, {}, {}
        for ann in instances['annotations']:
            Anns[ann['id']] = ann
            iid_to_raw_ann_ids[ann['image_id']] = iid_to_raw_ann_ids.get(
                    ann['image_id'], []) + [ann['id']]
        for img in instances['images']:
            Imgs[img['id']] = img
        
        # Make iid_to_ann_ids for this dataset
        img_ids = list(Imgs.keys())
        for img_id in tqdm(img_ids):
            if img_id in iid_to_ann_ids:
                continue
            raw_ann_ids = iid_to_raw_ann_ids[img_id]
            # raw_gd_bb
            raw_gd_bb = np.array([Anns[ann_id]['bbox'] 
                                for ann_id in raw_ann_ids]) # (n, 4) xywh
            # denorm_bb
            im_width = Imgs[img_id]['width']
            im_height = Imgs[img_id]['height']
            img_feat = np.load(osp.join(args.feats_dir, 
                            f'visual_grounding_coco_gt_{int(img_id):012}.npz'))
            norm_bb = img_feat['norm_bb']
            x1, x2 = norm_bb[:, 0] * im_width, norm_bb[:, 2] * im_width
            y1, y2 = norm_bb[:, 1] * im_height, norm_bb[:, 3] * im_height
            w, h = norm_bb[:, 4] * im_width, norm_bb[:, 5] * im_height
            denorm_bb = np.stack([x1, y1, w, h], axis=1)  # (n,4)
            # re-order ann_ids
            ordered_ann_ids = recover_ann_ids(denorm_bb, raw_gd_bb, raw_ann_ids)
            # check difference
            ordered_gd_bb = np.array([Anns[ann_id]['bbox'] 
                              for ann_id in ordered_ann_ids]) # (n, 4)
            for i in range(denorm_bb.shape[0]):
                assert np.sum(np.abs(denorm_bb[i]-ordered_gd_bb[i])) < 0.01, \
                '%s, %s' %(denorm_bb[i], ordered_gd_bb[i])
            # check ann_ids set
            if set(ordered_ann_ids) != set(raw_ann_ids):
                print('Please check img_id[%s]'%img_id)
                warning_img_ids.add(img_id)
            # check length of ann_ids
            assert len(ordered_ann_ids) == len(raw_ann_ids)
            # add to iid_to_ann_ids
            iid_to_ann_ids[img_id] = ordered_ann_ids

    print('%s images contain dupicated bounding boxes.' % len(warning_img_ids))
    pprint(list(warning_img_ids))

    # save
    output_file = osp.join(args.output_dir, 'iid_to_ann_ids.json')
    with open(output_file, 'w') as f:
        json.dump({'iid_to_ann_ids': iid_to_ann_ids}, f)
    print('%s iid_to_ann_ids saved in %s.' % (len(iid_to_ann_ids), output_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refer_dir', 
                    default='data', 
                    help='folder saving all downloaded refer datasets')
    parser.add_argument('--feats_dir', 
                    default='cache/feats/visual_grounding_coco_gt',
                    help='folder saving butd features.')
    parser.add_argument('--output_dir', 
                    default='cache',
                    help='output folder saving img_id --> [ann_id]')
    args = parser.parse_args()
    main(args)