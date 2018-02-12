from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# add matplotlib before cv2, otherwise bug
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.rcParams['image.interpolation'] = 'nearest'

from scipy.misc import imread, imresize
import scipy.ndimage
import numpy as np
import argparse
import json
import os
import os.path as osp
import time
from pprint import pprint

import _init_paths
from layers.joint_match import JointMatching
from mrcn import inference
from model.nms_wrapper import nms
from utils.mask_utils import recover_masks

# torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable

# root directory
ROOT_DIR = osp.join(osp.dirname(__file__), '..')

# box functions
def xywh_to_xyxy(boxes):
  """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
  return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))

def show_boxes(img, boxes, colors, texts=None):
  # boxes [[xyxy]]
  plt.imshow(img)
  ax = plt.gca()
  for k in range(boxes.shape[0]):
    box = boxes[k]
    xmin, ymin, xmax, ymax = list(box)
    coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
    color = colors[k]
    linestyle = 'dashed' if color in ['yellow', 'blue', 'red'] else 'solid'
    ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2, linestyle=linestyle))
    if texts is not None:
      ax.text(xmin, ymin, texts[k], bbox={'facecolor':color, 'alpha':0.5})
    ax.set_axis_off()
    ax.set_yticks([])
    ax.set_xticks([])

def show_mask(img, mask, color):
  plt.imshow(img)
  ax = plt.gca()
  # show pred rle
  img = np.zeros((mask.shape[0], mask.shape[1], 3))
  for i in range(3): img[:,:,i] = color[i]
  ax.imshow(np.dstack([img, mask*255*0.6]).astype(np.uint8))
  ax.set_axis_off()
  ax.set_yticks([])
  ax.set_xticks([])


# MattNet instance
class MattNet():

  def __init__(self, args):
    # load model
    model_prefix = osp.join(ROOT_DIR, 'output', args.dataset+'_'+args.splitBy, args.model_id)
    tic = time.time()
    infos = json.load(open(model_prefix+'.json'))
    model_path = model_prefix + '.pth'
    self.dataset = args.dataset
    self.model_opt = infos['opt']
    self.word_to_ix = infos['word_to_ix']
    self.ix_to_att = {ix: att for att, ix in infos['att_to_ix'].items()}
    self.model = self.load_matnet_model(model_path, self.model_opt)
    print('MatNet [%s_%s\'s %s] loaded in %.2f seconds.' % \
          (args.dataset, args.splitBy, args.model_id, time.time()-tic))

    # load mask r-cnn 
    tic = time.time()
    args.imdb_name = self.model_opt['imdb_name']
    args.net_name = self.model_opt['net_name']
    args.tag = self.model_opt['tag']
    args.iters = self.model_opt['iters']
    self.mrcn = inference.Inference(args)
    self.imdb = self.mrcn.imdb
    print('Mask R-CNN: imdb[%s], tag[%s], id[%s_mask_rcnn_iter_%s] loaded in %.2f seconds.' % \
          (args.imdb_name, args.tag, args.net_name, args.iters, time.time()-tic))

  def load_matnet_model(self, checkpoint_path, opt):
    # load MatNet model from pre-trained checkpoint_path
    tic = time.time()
    model = JointMatching(opt)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'].state_dict())
    model.eval()
    model.cuda()
    return model

  def forward_image(self, img_path, nms_thresh=.3, conf_thresh=.65):
    """
    Arguments:
    - img_path   : path to image
    - nms_thresh : nms threshold 
    - conf_thresh: confidence threshold [0,1]
    Return "data" is a dict of
    - det_ids: list of det_ids, order consistent with dets and masks
    - dets   : [{det_id, box, category_name, category_id, score}], box is [xywh] and category_id is coco_cat_id
    - masks  : ndarray (n, im_h, im_w) uint8 [0,1]
    - Feats  :
      - pool5     : Variable cuda (n, 1024, 7, 7)
      - fc7       : Variable cuda (n, 2048, 7, 7)
      - lfeats    : Variable cuda (n, 5)
      - dif_lfeats: Variable cuda (n, 5*topK)
      - cxt_fc7   : Variable cuda (n, topK, 2048)
      - cxt_lfeats: Variable cuda (n, topK, 5)
    - cxt_det_ids : list of [surrounding_det_ids] for each det_id
    """
    # read image
    im = imread(img_path)

    # 1st step: detect objects
    scores, boxes = self.mrcn.predict(img_path)

    # get head feats, i.e., net_conv 
    net_conv = self.mrcn.net._predictions['net_conv']  # Variable cuda (1, 1024, h, w)
    im_info = self.mrcn.net._im_info  # [[H, W, im_scale]]

    # get cls_to_dets, class_name -> [xyxys] which is (n, 5)
    cls_to_dets, num_dets = self.cls_to_detections(scores, boxes, nms_thresh, conf_thresh)
    # make sure num_dets > 0
    thresh = conf_thresh
    while num_dets == 0:
      thresh -= 0.1
      cls_to_dets, num_dets = self.cls_to_detections(scores, boxes, nms_thresh, thresh)

    # add to dets
    dets = []
    det_id = 0
    for category_name, detections in cls_to_dets.items():
      # detections: list of (n, 5), [xyxyc]
      for detection in detections:
        x1, y1, x2, y2, sc = detection
        det = {'det_id': det_id, 
               'box': [x1, y1, x2-x1+1, y2-y1+1],
               'category_name': category_name,
               'category_id': self.imdb._class_to_coco_cat_id[category_name],
               'score': sc}
        dets += [det]
        det_id += 1
    Dets = {det['det_id']: det for det in dets}
    det_ids = [det['det_id'] for det in dets]

    # 2nd step: get masks
    boxes = xywh_to_xyxy(np.array([det['box'] for det in dets]))  # xyxy (n, 4) ndarray
    labels = np.array([self.imdb._class_to_ind[det['category_name']] for det in dets])
    mask_prob = self.mrcn.net._predict_masks_from_boxes_and_labels(net_conv, boxes*im_info[0][2], labels)
    mask_prob = mask_prob.data.cpu().numpy()
    masks = recover_masks(mask_prob, boxes, im.shape[0], im.shape[1])  # (N, ih, iw) uint8 [0-255]
    masks = (masks > 122.).astype(np.uint8)  # (N, ih, iw) uint8 [0,1]

    # 3rd step: compute features
    pool5, fc7 = self.mrcn.box_to_spatial_fc7(net_conv, im_info, boxes)  # (n, 1024, 7, 7), (n, 2048, 7, 7)
    lfeats = self.compute_lfeats(det_ids, Dets, im)
    dif_lfeats = self.compute_dif_lfeats(det_ids, Dets)
    cxt_fc7, cxt_lfeats, cxt_det_ids = self.fetch_cxt_feats(det_ids, Dets, fc7, self.model_opt)

    # move to Variable cuda
    lfeats = Variable(torch.from_numpy(lfeats).cuda())
    dif_lfeats = Variable(torch.from_numpy(dif_lfeats).cuda())
    cxt_lfeats = Variable(torch.from_numpy(cxt_lfeats).cuda())

    # return
    data = {}
    data['det_ids'] = det_ids
    data['dets'] = dets
    data['masks'] = masks
    data['cxt_det_ids'] = cxt_det_ids
    data['Feats'] = {'pool5': pool5, 'fc7': fc7, 'lfeats': lfeats, 'dif_lfeats': dif_lfeats, 
                     'cxt_fc7': cxt_fc7, 'cxt_lfeats': cxt_lfeats}
    return data

  def comprehend(self, img_data, expr):
    """
    Arguments:
    - img_data: computed from self.forward_image()
    - expr    : expression in string format
    Return entry is a dict of:
    - tokens     : list of words
    - pred_det_id: predicted det_id
    - pred_box   : pred_det's box [xywh]
    - rel_det_id : relative det_id
    - rel_box    : relative box [xywh]
    - sub_grid_attn: list of 49 attn
    - sub_attn   : list of seq_len attn
    - loc_attn   : list of seq_len attn
    - rel_attn   : list of seq_len attn
    - weights    : list of 3 module weights
    - pred_atts  : top 5 attributes, list of (att_wd, score)
    """
    # image data
    det_ids = img_data['det_ids']
    cxt_det_ids = img_data['cxt_det_ids']
    Dets = {det['det_id']: det for det in  img_data['dets']}
    masks = img_data['masks']
    Feats = img_data['Feats']

    # encode labels
    expr = expr.lower().strip()
    labels = self.encode_labels([expr], self.word_to_ix)  # (1, sent_length)
    labels = Variable(torch.from_numpy(labels).long().cuda())
    expanded_labels = labels.expand(len(det_ids), labels.size(1)) # (n, sent_length)
    scores, sub_grid_attn, sub_attn, loc_attn, rel_attn, rel_ixs, weights, att_scores = \
          self.model(Feats['pool5'], Feats['fc7'], Feats['lfeats'], Feats['dif_lfeats'], 
                     Feats['cxt_fc7'], Feats['cxt_lfeats'], 
                     expanded_labels) 

    # move to numpy
    scores = scores.data.cpu().numpy()
    pred_ix = np.argmax(scores)
    pred_det_id = det_ids[pred_ix]
    att_scores = F.sigmoid(att_scores)  # (n, num_atts)
    rel_ixs = rel_ixs.data.cpu().numpy().tolist()  # (n, )
    rel_ix = rel_ixs[pred_ix]

    # get everything
    entry = {}
    entry['tokens'] = expr.split()
    entry['pred_det_id'] = det_ids[pred_ix]
    entry['pred_box'] = Dets[pred_det_id]['box']
    entry['pred_mask'] = masks[pred_ix]
    # relative det_id
    entry['rel_det_id'] = cxt_det_ids[pred_ix][rel_ix]
    entry['rel_box'] = Dets[entry['rel_det_id']]['box'] if entry['rel_det_id'] > 0 else [0,0,0,0]
    # attention
    entry['sub_grid_attn'] = sub_grid_attn[pred_ix].data.cpu().numpy().tolist() # list of 49 attn
    entry['sub_attn'] = sub_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
    entry['loc_attn'] = loc_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
    entry['rel_attn'] = rel_attn[pred_ix].data.cpu().numpy().tolist() # list of seq_len attn
    entry['weights'] = weights[pred_ix].data.cpu().numpy().tolist()   # list of 3 weights
    # attributes
    pred_atts = []  # list of (att_wd, score)
    pred_att_scores = att_scores[pred_ix].data.cpu().numpy()
    top_att_ixs = pred_att_scores.argsort()[::-1][:5] # check top 5 attributes
    for k in top_att_ixs:
        pred_atts.append((self.ix_to_att[k], float(pred_att_scores[k])))
    entry['pred_atts'] = pred_atts

    return entry


  def encode_labels(self, sent_str_list, word_to_ix):
    """
    Arguments:
    - sent_str_list: list of n sents in string format
    return:
    - labels: int32 (n, sent_length)
    """
    num_sents = len(sent_str_list)
    max_len = max([len(sent_str.split()) for sent_str in sent_str_list])
    L = np.zeros((num_sents, max_len), dtype=np.int32)
    for i, sent_str in enumerate(sent_str_list):
      tokens = sent_str.split()
      for j, w in enumerate(tokens):
        L[i,j] = word_to_ix[w] if w in word_to_ix else word_to_ix['<UNK>']
    return L

  def compute_lfeats(self, det_ids, Dets, im):
    # Compute (n, 5) lfeats for given det_ids
    lfeats = np.empty((len(det_ids), 5), dtype=np.float32)
    for ix, det_id in enumerate(det_ids):
      det = Dets[det_id]
      x,y,w,h = det['box']
      ih, iw = im.shape[0], im.shape[1]
      lfeats[ix] = np.array([[x/iw, y/ih, (x+w-1)/iw, (y+h-1)/ih, w*h/(iw*ih)]], np.float32) 
    return lfeats


  def fetch_neighbour_ids(self, ref_det_id, Dets):
    """
    For a given ref_det_id, we return
    - st_det_ids: same-type neighbouring det_ids (not including itself)
    - dt_det_ids: different-type neighbouring det_ids
    Ordered by distance to the input det_id
    """
    ref_det = Dets[ref_det_id]
    x, y, w, h = ref_det['box']
    rx, ry = x+w/2, y+h/2

    def compare(det_id0, det_id1):
      x, y, w, h = Dets[det_id0]['box']
      ax0, ay0 = x+w/2, y+h/2
      x, y, w, h = Dets[det_id1]['box']
      ax1, ay1 = x+w/2, y+h/2
      # closer --> former
      if (rx-ax0)**2 + (ry-ay0)**2 <= (rx-ax1)**2 + (ry-ay1)**2:
        return -1
      else:
        return 1
        
    det_ids = list(Dets.keys())  # copy in case the raw list is changed
    det_ids = sorted(det_ids, cmp=compare)
    st_det_ids, dt_det_ids = [], []
    for det_id in det_ids:
      if det_id != ref_det_id:
        if Dets[det_id]['category_id'] == ref_det['category_id']:
          st_det_ids += [det_id]
        else:
          dt_det_ids += [det_id]
    return st_det_ids, dt_det_ids  


  def compute_dif_lfeats(self, det_ids, Dets, topK=5):
    # return ndarray float32 (#det_ids, 5*topK)
    dif_lfeats = np.zeros((len(det_ids), 5*topK), dtype=np.float32)
    for i, ref_det_id in enumerate(det_ids):
      # reference box 
      rbox = Dets[ref_det_id]['box']
      rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
      # candidate boxes
      st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id, Dets)
      for j, cand_det_id in enumerate(st_det_ids[:topK]):
        cbox = Dets[cand_det_id]['box']
        cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
        dif_lfeats[i, j*5:(j+1)*5] = \
            np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
    return dif_lfeats


  def fetch_cxt_feats(self, det_ids, Dets, spatial_fc7, opt):
    """
    Arguments:
    - det_ids    : list of det_ids
    - Dets       : each det is {det_id, box, category_id, category_name}
    - spatial_fc7: (#det_ids, 2048, 7, 7) Variable cuda 
    Return 
    - cxt_feats  : Variable cuda (#det_ids, topK, feat_dim)
    - cxt_lfeats : ndarray (#det_ids, topK, 5) 
    - cxt_det_ids: [[det_id]] of size (#det_ids, topK), padded with -1
    Note we use neighbouring objects for computing context objects, zeros padded.    
    """
    fc7 = spatial_fc7.mean(3).mean(2)  # (n, 2048)
    topK = opt['num_cxt']
    cxt_feats = Variable(spatial_fc7.data.new(len(det_ids), topK, 2048).zero_())
    cxt_lfeats = np.zeros((len(det_ids), topK, 5), dtype=np.float32)
    cxt_det_ids = -np.ones((len(det_ids), topK), dtype=np.int32) # (#det_ids, topK)
    for i, ref_det_id in enumerate(det_ids):
      # reference box
      rbox = Dets[ref_det_id]['box']
      rcx, rcy, rw, rh = rbox[0]+rbox[2]/2, rbox[1]+rbox[3]/2, rbox[2], rbox[3]
      # candidate boxes
      st_det_ids, dt_det_ids = self.fetch_neighbour_ids(ref_det_id, Dets)
      if opt['with_st'] > 0:
        cand_det_ids = dt_det_ids + st_det_ids
      else:
        cand_det_ids = dt_det_ids
      cand_det_ids = cand_det_ids[:topK]
      for j, cand_det_id in enumerate(cand_det_ids):
        cand_det = Dets[cand_det_id]
        cbox = cand_det['box']
        cx1, cy1, cw, ch = cbox[0], cbox[1], cbox[2], cbox[3]
        cxt_lfeats[i, j, :] = np.array([(cx1-rcx)/rw, (cy1-rcy)/rh, (cx1+cw-rcx)/rw, (cy1+ch-rcy)/rh, cw*ch/(rw*rh)])
        cxt_feats[i, j, :] = fc7[det_ids.index(cand_det_id)]
        cxt_det_ids[i, j] = cand_det_id
    cxt_det_ids = cxt_det_ids.tolist()
    return cxt_feats, cxt_lfeats, cxt_det_ids


  def cls_to_detections(self, scores, boxes, nms_thresh, conf_thresh):
      # run nms and threshold for each class detection
      cls_to_dets = {}
      num_dets = 0
      for cls_ind, class_name in enumerate(self.imdb.classes[1:]):
          cls_ind += 1  # because we skipped background
          cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind+1)]
          cls_scores = scores[:, cls_ind]
          dets = np.hstack((cls_boxes,
                            cls_scores[:, np.newaxis])).astype(np.float32)
          keep = nms(torch.from_numpy(dets), nms_thresh)
          dets = dets[keep.numpy(), :]
          inds = np.where(dets[:, -1] >= conf_thresh)[0]
          dets = dets[inds, :]
          cls_to_dets[class_name] = dets
          num_dets += dets.shape[0]
      return cls_to_dets, num_dets


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, default='refcoco', help='dataset name: refclef, refcoco, refcoco+, refcocog')
  parser.add_argument('--splitBy', type=str, default='unc', help='splitBy: unc, google, berkeley')
  parser.add_argument('--model_id', type=str, default='mrcn_cmr_with_st', help='model id name')
  args = parser.parse_args()


