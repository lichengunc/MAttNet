import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch 
import torch.nn.functional as F

# IoU function
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
      inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
      inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union

def eval_split(matt_dataset, model, opt):
    verbose = opt.get('verbose', True)
    num_sents = opt.get('num_sents', -1)

    # set mode
    model.eval()

    # initialize
    num_evals = 0
    acc = 0
    predictions = []
    finish_flag = False
    model_time = 0

    for i, image_id in enumerate(matt_dataset.split_image_ids):

        data = matt_dataset.getImageBatch(image_id)
        image_det_ids = data['det_ids']  # n det_ids
        sent_ids = data['sent_ids']  # m sent_id
        gd_boxes = data['gd_boxes']  # m [xywh]
        labels = data['labels'].cuda()  
        Feats = {_k: data['Feats'][_k].cuda() for _k in data['Feats']}
        
        for j, sent_id in enumerate(sent_ids):

            # expand labels
            label = labels[j:j+1]      # (1, label.size(1))
            max_len = (label != 0).sum().item()
            label = label[:, :max_len] # (1, max_len) 
            expanded_labels = label.expand(
                                len(image_det_ids), max_len) # (n, max_len)

            # forward
            tic = time.time()
            scores, sub_attn, loc_attn, rel_attn, rel_ixs, weights = \
                model(Feats['vfeats'], Feats['lfeats'], Feats['dif_lfeats'], 
                      Feats['cxt_vfeats'], Feats['cxt_lfeats'], expanded_labels)
            scores = scores.data.cpu().numpy()
            rel_ixs = rel_ixs.data.cpu().numpy().tolist()

            # compute acc 
            pred_ix = np.argmax(scores)
            pred_det_id = image_det_ids[pred_ix]
            pred_box = matt_dataset.Dets[pred_det_id]['box']  # xywh
            gd_box = gd_boxes[j]
            if computeIoU(pred_box, gd_box) >= 0.5:
                acc += 1
            num_evals += 1

            # relative det_id
            rel_ix = rel_ixs[pred_ix]

            # add info
            entry = {
                'sent_id': sent_id,
                'sent': matt_dataset.decode_labels(label.data.cpu().numpy())[0],
                'gd_box': gd_box,
                'pred_box': pred_box,
                'pred_score': scores.tolist()[pred_ix],
                'sub_attn': sub_attn[pred_ix].data.cpu().numpy().tolist(), 
                'loc_attn': loc_attn[pred_ix].data.cpu().numpy().tolist(), 
                'rel_attn': rel_attn[pred_ix].data.cpu().numpy().tolist(), 
                'rel_det_id': data['cxt_det_ids'][pred_ix][rel_ix],        
                'weights': weights[pred_ix].data.cpu().numpy().tolist()
            } 
            predictions.append(entry)
            model_time += (time.time() - tic)

        # print
        if verbose and (i+1) % 100 == 0:
            print(f'Evaluating [{matt_dataset.split}] '
                  f'image[{i+1}/{len(matt_dataset)}]\'s sents, '
                  f'acc={acc*100./num_evals:.2f}%, '
                  f'model time (per sent) is {model_time/len(sent_ids):.2f}s.')
        model_time = 0
    
    return acc/num_evals, predictions