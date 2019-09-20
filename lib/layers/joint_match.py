import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.lang_encoder import RNNEncoder, PhraseAttention
from layers.visual_encoder import (LocationEncoder, RelationEncoder)

"""
Simple Matching function for
- visual_input (n, vis_dim)  
- lang_input (n, vis_dim)
forward them through several mlp layers and finally inner-product, get cossim
"""
class Matching(nn.Module):
    
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_drop_out):
        super(Matching, self).__init__()
        self.vis_emb_fc  = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_drop_out),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        )
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_drop_out),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim)
                                        ) 

    def forward(self, visual_input, lang_input):
        """
        Inputs:
        - visual_input: float32 (n, vis_dim)
        - lang_input  : float32 (n, lang_dim)
        Output:
        - cossim      : float32 (n, 1), which is inner-product of two views
        """
        # forward two views
        visual_emb = self.vis_emb_fc(visual_input)
        lang_emb = self.lang_emb_fc(lang_input)

        # l2-normalize (n, jemb_dim)
        visual_emb_normalized = nn.functional.normalize(visual_emb, p=2, dim=1) 
        lang_emb_normalized = nn.functional.normalize(lang_emb, p=2, dim=1)

        # compute cossim
        cossim = torch.sum(
                    visual_emb_normalized * lang_emb_normalized, 1) # (n, )
        return cossim.view(-1, 1)  # (n, 1)

"""
Relation Matching function for
- visual_input (n, m, vis_dim)  
- lang_input   (n, vis_dim)
- masks        (n, m) 
forward them through several mlp layers and finally inner-product, 
and get cossim (n, )
"""
class RelationMatching(nn.Module):
    def __init__(self, vis_dim, lang_dim, jemb_dim, jemb_drop_out):
        super(RelationMatching, self).__init__()
        self.lang_dim = lang_dim
        self.vis_emb_fc  = nn.Sequential(nn.Linear(vis_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_drop_out),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        )
        self.lang_emb_fc = nn.Sequential(nn.Linear(lang_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim),
                                        nn.ReLU(),
                                        nn.Dropout(jemb_drop_out),
                                        nn.Linear(jemb_dim, jemb_dim),
                                        nn.BatchNorm1d(jemb_dim)
                                        ) 

    def forward(self, visual_input, lang_input, masks):
        """Inputs:
        - visual_input : (n, m, vis_dim)
        - lang_input   : (n, lang_dim)
        - masks        : (n, m)
        Output:
        - cossim       : (n, 1)
        """
        # forward two views
        n, m = visual_input.size(0), visual_input.size(1)
        visual_emb = self.vis_emb_fc(visual_input.view(n*m, -1)) #(nm, jemb_dim)
        lang_input = lang_input.unsqueeze(1).expand(
                        n, m, self.lang_dim).contiguous() # (n, m, lang_dim)
        lang_input = lang_input.view(n*m, -1)      # (n x m, lang_dim)
        lang_emb   = self.lang_emb_fc(lang_input)  # (n x m, jemb_dim)

        # l2-normalize
        visual_emb_normalized = nn.functional.normalize(visual_emb, p=2, dim=1)
        lang_emb_normalized   = nn.functional.normalize(lang_emb, p=2, dim=1)   

        # compute cossim
        cossim = torch.sum(
            visual_emb_normalized * lang_emb_normalized, 1)  # (nxm, )
        cossim = cossim.view(n, m)    # (n, m)
        # mask cossim
        cossim = masks * cossim       # (n, m)
        # pick max
        cossim, ixs = torch.max(cossim, 1) # (n, ), (n, )
        
        return cossim.view(-1, 1), ixs

class JointMatching(nn.Module):

    def __init__(self, opt):
        super(JointMatching, self).__init__()
        num_layers = opt['rnn_num_layers']
        hidden_size = opt['rnn_hidden_size']
        num_dirs = 2 if opt['bidirectional'] > 0 else 1
        jemb_dim = opt['jemb_dim']

        # language rnn encoder
        self.rnn_encoder = RNNEncoder(
                                vocab_size=opt['vocab_size'],
                                word_embedding_size=opt['word_embedding_size'],
                                word_vec_size=opt['word_vec_size'],
                                hidden_size=opt['rnn_hidden_size'],
                                bidirectional=opt['bidirectional']>0,
                                input_dropout_p=opt['word_drop_out'],
                                dropout_p=opt['rnn_drop_out'],
                                n_layers=opt['rnn_num_layers'],
                                rnn_type=opt['rnn_type'],
                                variable_lengths=opt['variable_lengths']>0
                            )

        # [vis; loc] weighter
        self.weight_fc = nn.Linear(num_layers * num_dirs * hidden_size, 3)

        # phrase attender
        self.sub_attn = PhraseAttention(hidden_size * num_dirs)
        self.loc_attn = PhraseAttention(hidden_size * num_dirs)
        self.rel_attn = PhraseAttention(hidden_size * num_dirs)

        # visual matching 
        self.sub_matching = Matching(opt['fc7_dim'], opt['word_vec_size'], 
                                    opt['jemb_dim'], opt['jemb_drop_out'])

        # location matching
        self.loc_encoder = LocationEncoder(opt)
        self.loc_matching = Matching(opt['jemb_dim'], opt['word_vec_size'], 
                                     opt['jemb_dim'], opt['jemb_drop_out'])

        # relation matching
        self.rel_encoder  = RelationEncoder(opt)
        self.rel_matching = RelationMatching(
                                opt['jemb_dim'], opt['word_vec_size'],
                                opt['jemb_dim'], opt['jemb_drop_out']) 

    def forward(self, vfeats, lfeats, dif_lfeats, 
                cxt_vfeats, cxt_lfeats, labels):
        """
        Inputs:
        - vfeats      : (n, fc7_dim)
        - lfeats      : (n, 5)
        - dif_lfeats  : (n, 25)
        - cxt_vfeats  : (n, 5, fc7_dim)
        - cxt_lfeats  : (n, 5, 5)
        - labels      : (n, seq_len)
        Output:
        - scores        : (n, )
        - sub_attn      : (n, seq_len) attn on subjective words of expression 
        - loc_attn      : (n, seq_len) attn on location words of expression
        - rel_attn      : (n, seq_len) attn on relation words of expression
        - rel_ixs       : (n, ) selected context object
        - weights       : (n, 3) attn on modules
        """
        # expression encoding
        context, hidden, embedded = self.rnn_encoder(labels)

        # weights on [sub; loc]
        weights = F.softmax(self.weight_fc(hidden), dim=1) # (n, 3)

        # subject matching
        sub_attn, sub_phrase_emb = self.sub_attn(context, embedded, labels)
        sub_matching_scores = self.sub_matching(vfeats, sub_phrase_emb) # (n, 1)

        # location matching
        loc_attn, loc_phrase_emb = self.loc_attn(context, embedded, labels)
        loc_feats = self.loc_encoder(lfeats, dif_lfeats)  # (n, 512)
        loc_matching_scores = self.loc_matching(
                                loc_feats, loc_phrase_emb)  # (n, 1)

        # relationship matching
        rel_attn, rel_phrase_emb = self.rel_attn(context, embedded, labels)
        rel_feats, masks = self.rel_encoder(
                    cxt_vfeats, cxt_lfeats)  # (n, num_cxt, 512), (n, num_cxt)
        rel_matching_scores, rel_ixs = self.rel_matching(
                    rel_feats, rel_phrase_emb, masks) # (n, 1), (n, )

        # final scores
        scores = (weights * torch.cat([sub_matching_scores, 
                                       loc_matching_scores, 
                                       rel_matching_scores], 1)).sum(1) # (n, )

        return scores, sub_attn, loc_attn, rel_attn, rel_ixs, weights
