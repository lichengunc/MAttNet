from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Normalize_Scale(nn.Module):
  def __init__(self, dim, init_norm=20):
    super(Normalize_Scale, self).__init__()
    self.init_norm = init_norm
    self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

  def forward(self, bottom):
    # input is variable (n, dim)
    assert isinstance(bottom, Variable), 'bottom must be variable'
    bottom_normalized = nn.functional.normalize(bottom, p=2, dim=1)
    bottom_normalized_scaled = bottom_normalized * self.weight
    return bottom_normalized_scaled

"""
Takes lfeats (n, 5) and dif_lfeats (n, 25) as inputs, then
output fused location features (n, 512)
"""
class LocationEncoder(nn.Module):
  def __init__(self, opt):
    super(LocationEncoder, self).__init__()
    init_norm = opt.get('visual_init_norm', 20)
    self.lfeat_normalizer = Normalize_Scale(5, init_norm)
    self.dif_lfeat_normalizer = Normalize_Scale(25, init_norm)
    self.fc = nn.Linear(5+25, opt['jemb_dim'])

  def forward(self, lfeats, dif_lfeats):
    concat = torch.cat([self.lfeat_normalizer(lfeats), self.dif_lfeat_normalizer(dif_lfeats)], 1)
    output = self.fc(concat)
    return output

"""
Takes ann_feats (n, visual_feat_dim, 49) and phrase_emb (n, word_vec_size)
output attended visual feats (n, visual_feat_dim) and attention (n, 49)
Equations:
 vfeats = vemb(ann_feats)  # extract useful and abstract info (instead of each grid feature)
 hA = tanh(W([vfeats, P]))
 attn = softmax(W hA +b)   # compute phrase-conditioned attention
 weighted_vfeats = attn.*vfeats
 output = L([obj_feats, weighted_vfeats])  # (n, jemb_dim)
"""
class SubjectEncoder(nn.Module):

  def __init__(self, opt):
    super(SubjectEncoder, self).__init__()
    self.word_vec_size = opt['word_vec_size']
    self.jemb_dim = opt['jemb_dim']
    self.pool5_dim, self.fc7_dim = opt['pool5_dim'], opt['fc7_dim']
    self.pool5_normalizer = Normalize_Scale(opt['pool5_dim'], opt['visual_init_norm'])
    self.fc7_normalizer   = Normalize_Scale(opt['fc7_dim'],   opt['visual_init_norm'])
    self.att_normalizer   = Normalize_Scale(opt['jemb_dim'],  opt['visual_init_norm'])
    self.phrase_normalizer= Normalize_Scale(opt['word_vec_size'], opt['visual_init_norm'])
    self.att_fuse = nn.Sequential(nn.Linear(opt['pool5_dim']+opt['fc7_dim'], opt['jemb_dim']),
                                  nn.BatchNorm1d(opt['jemb_dim']), 
                                  nn.ReLU())
    self.att_dropout = nn.Dropout(opt['visual_drop_out'])
    self.att_fc = nn.Linear(opt['jemb_dim'], opt['num_atts'])

    self.attn_fuse = nn.Sequential(nn.Linear(opt['fc7_dim']+opt['jemb_dim']+opt['word_vec_size'], opt['jemb_dim']),
                                   nn.Tanh(),
                                   nn.Linear(opt['jemb_dim'], 1))


  def forward(self, pool5, fc7, phrase_emb):
    """Inputs
    - pool5     : (n, 1024, 7, 7)
    - fc7       : (n, 2048, 7, 7)
    - phrase_emb: (n, word_vec_size)
    Outputs
    - visual_out: (n, fc7_dim + att_dim)
    - attn      : (n, 49)
    - att_scores: (n, num_atts)
    """
    batch, grids = pool5.size(0), pool5.size(2)*pool5.size(3)

    # normalize and reshape pool5 & fc7
    pool5 = pool5.view(batch, self.pool5_dim, -1) # (n, 1024, 49)
    pool5 = pool5.transpose(1,2).contiguous().view(-1, self.pool5_dim) # (nx49, 1024)
    pool5 = self.pool5_normalizer(pool5) # (nx49, 1024)
    fc7 = fc7.view(batch, self.fc7_dim, -1)  # (n, 2048, 49)
    fc7 = fc7.transpose(1,2).contiguous().view(-1, self.fc7_dim) # (n x 49, 2048)
    fc7 = self.fc7_normalizer(fc7) # (nx49, 2048)

    # att_feats   
    att_feats = self.att_fuse(torch.cat([pool5, fc7], 1)) # (nx49, 512)

    # predict atts
    avg_att_feats = att_feats.view(batch, -1, self.jemb_dim).mean(1) # (n, 512)
    avg_att_feats = self.att_dropout(avg_att_feats) # dropout
    att_scores = self.att_fc(avg_att_feats) # (n, num_atts)

    # compute spatial attention
    att_feats = self.att_normalizer(att_feats)     # (nx49, 512)
    visual_feats = torch.cat([fc7, att_feats], 1)  # (nx49, 2048+512)
    phrase_emb = self.phrase_normalizer(phrase_emb)# (n, word_vec_size) 
    phrase_emb = phrase_emb.unsqueeze(1).expand(batch, grids, self.word_vec_size) # (n, 49, word_vec_size)
    phrase_emb = phrase_emb.contiguous().view(-1, self.word_vec_size) # (nx49, word_vec_size) 
    attn = self.attn_fuse(torch.cat([visual_feats, phrase_emb], 1)) # (nx49, 1)
    attn = F.softmax(attn.view(batch, grids)) # (n, 49)

    # weighted sum
    attn3 = attn.unsqueeze(1)  # (n, 1, 49)
    weighted_visual_feats = torch.bmm(attn3, visual_feats.view(batch, grids, -1)) # (n, 1, 2048+512)
    weighted_visual_feats = weighted_visual_feats.squeeze(1) # (n, 2048+512)

    return weighted_visual_feats, attn, att_scores

  def extract_subj_feats(self, pool5, fc7):
    """Inputs
    - pool5     : (n, 1024, 7, 7)
    - fc7       : (n, 2048, 7, 7)
    Outputs
    - visual_out: (n, fc7_dim + att_dim)
    - att_scores: (n, num_atts)
    """
    batch, grids = pool5.size(0), pool5.size(2)*pool5.size(3)

    # normalize and reshape pool5 & fc7
    pool5 = pool5.view(batch, self.pool5_dim, -1) # (n, 1024, 49)
    pool5 = pool5.transpose(1,2).contiguous().view(-1, self.pool5_dim) # (nx49, 1024)
    pool5 = self.pool5_normalizer(pool5) # (nx49, 1024)
    fc7 = fc7.view(batch, self.fc7_dim, -1)  # (n, 2048, 49)
    fc7 = fc7.transpose(1,2).contiguous().view(-1, self.fc7_dim) # (n x 49, 2048)
    fc7 = self.fc7_normalizer(fc7) # (nx49, 2048)

    # att_feats   
    att_feats = self.att_fuse(torch.cat([pool5, fc7], 1)) # (nx49, 512)

    # predict atts
    avg_att_feats = att_feats.view(batch, -1, self.jemb_dim).mean(1) # (n, 512)
    avg_att_feats = self.att_dropout(avg_att_feats) # dropout
    att_scores = self.att_fc(avg_att_feats) # (n, num_atts)

    # compute spatial attention
    att_feats = self.att_normalizer(att_feats)     # (nx49, 512)
    visual_feats = torch.cat([fc7, att_feats], 1)  # (nx49, 2048+512)

    return visual_feats, att_scores

"""
Takes relative location (n, c, 5) and object features (n, c, 2048) as inputs, then
output encoded contexts (n, c, 512) and masks (n, c)
"""
class RelationEncoder(nn.Module):
  def __init__(self, opt):
    super(RelationEncoder, self).__init__()
    self.vis_feat_normalizer = Normalize_Scale(opt['fc7_dim'], opt['visual_init_norm'])
    self.lfeat_normalizer    = Normalize_Scale(5, opt['visual_init_norm'])
    self.fc = nn.Linear(opt['fc7_dim'] + 5, opt['jemb_dim'])

  def forward(self, cxt_feats, cxt_lfeats):
    """Inputs:
    - cxt_feats : (n, num_cxt, fc7_dim)
    - cxt_lfeats: (n, num_cxt, 5)
    Return:
    - rel_feats : (n, num_cxt, jemb_dim)
    - masks     : (n, num_cxt)
    """
    # compute masks first
    masks = (cxt_lfeats.sum(2) != 0).float()  # (n, num_cxt)

    # compute joint encoded context
    batch, num_cxt = cxt_feats.size(0), cxt_feats.size(1)
    cxt_feats  = self.vis_feat_normalizer(cxt_feats.view(batch*num_cxt, -1)) # (batch * num_cxt, fc7_dim)
    cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.view(batch*num_cxt, -1))  # (batch * num_cxt, 5)

    # joint embed
    concat = torch.cat([cxt_feats, cxt_lfeats], 1) # (batch * num_cxt, fc7_dim + 5)
    rel_feats = self.fc(concat)                    # (batch * num_cxt, jemb_dim)
    rel_feats = rel_feats.view(batch, num_cxt, -1) # (batch, num_cxt, jemb_dim)

    # return
    return rel_feats, masks