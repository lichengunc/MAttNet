from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json
import h5py
import time
from pprint import pprint

import torch 
import torch.nn.functional as F
from torch.autograd import Variable

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


def eval_split(loader, model, crit, split, opt):
  verbose = opt.get('verbose', True)
  num_sents = opt.get('num_sents', -1)
  assert split != 'train', 'Check the evaluation split. (comment this line if you are evaluating [train])'

  # set mode
  model.eval()

  # initialize
  n = 0
  loss_evals = 0
  acc = 0
  predictions = []
  finish_flag = False

  while True:

    data = loader.getTestBatch(split, opt)      
    det_ids = data['det_ids']
    sent_ids = data['sent_ids']
    Feats = data['Feats'] 
    labels = data['labels']
        
    for i, sent_id in enumerate(sent_ids):

      # expand labels
      label = labels[i:i+1]      # (1, label.size(1))
      max_len = (label != 0).sum().data[0]
      label = label[:, :max_len] # (1, max_len) 
      expanded_labels = label.expand(len(det_ids), max_len) # (n, max_len)

      # forward
      # scores  : overall matching score (n, )
      # sub_grid_attn : (n, 49) attn on subjective's grids
      # sub_attn: (n, seq_len) attn on subjective words of expression
      # loc_attn: (n, seq_len) attn on location words of expression
      # rel_attn: (n, seq_len) attn on relation words of expression
      # rel_ixs : (n, ) selected context object
      # weights : (n, 2) weights on subj and loc
      # att_scores: (n, num_atts)
      scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights, att_scores = \
        model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'], 
              Feats['cxt_fc7'], Feats['cxt_lfeats'],
              expanded_labels) 
      scores = scores.data.cpu().numpy()
      att_scores = F.sigmoid(att_scores) # (n, num_atts)
      rel_ixs = rel_ixs.data.cpu().numpy().tolist() # (n, )

      # compute loss
      pred_ix = np.argmax(scores)
      pred_det_id = det_ids[pred_ix]
      pred_box = loader.Dets[pred_det_id]['box']
      gd_box = data['gd_boxes'][i]
      if computeIoU(pred_box, gd_box) >= 0.5:
        acc += 1
      loss_evals += 1

      # relative det_id
      rel_ix = rel_ixs[pred_ix]

      # predict attribute on the predicted object
      pred_atts = []
      pred_att_scores = att_scores[pred_ix].data.cpu().numpy()
      top_att_ixs = pred_att_scores.argsort()[::-1][:5] # check top 5 attributes
      for k in top_att_ixs:
        pred_atts.append((loader.ix_to_att[k], float(pred_att_scores[k])))
       
      # add info
      entry = {}
      entry['image_id'] = data['image_id']
      entry['sent_id'] = sent_id
      entry['sent'] = loader.decode_labels(label.data.cpu().numpy())[0] # gd-truth sent
      entry['gd_box'] = gd_box
      entry['pred_det_id'] = data['det_ids'][pred_ix]
      entry['pred_box'] = pred_box  
      entry['pred_score'] = scores.tolist()[pred_ix]
      entry['sub_grid_attn'] = sub_grid_attn[pred_ix].data.cpu().numpy().tolist() # list of 49 attn
      entry['sub_attn'] = sub_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
      entry['loc_attn'] = loc_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
      entry['rel_attn'] = rel_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
      entry['rel_det_id'] = data['cxt_det_ids'][pred_ix][rel_ix]        # rel det_id
      entry['rel_box'] = loader.Dets[entry['rel_det_id']]['box'] if entry['rel_det_id'] > 0 else [0,0,0,0]
      entry['weights'] = weights[pred_ix].data.cpu().numpy().tolist()   # list of 3 weights
      entry['pred_atts'] = pred_atts # list of (att_wd, score)
      predictions.append(entry)

      # if used up 
      if num_sents > 0 and loss_eval >= num_sents:
        finish_flag = True
        break

    # print
    ix0 = data['bounds']['it_pos_now']
    ix1 = data['bounds']['it_max']
    if verbose:
      print('evaluating [%s] ... image[%d/%d]\'s sents, acc=%.2f%%' % \
            (split, ix0, ix1, acc*100.0/loss_evals))
    
    # if we wrapped around the split
    if finish_flag or data['bounds']['wrapped']:
      break

  return acc/loss_evals, predictions

