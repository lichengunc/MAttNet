import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalize_Scale(nn.Module):
    def __init__(self, dim, init_norm=20):
        super(Normalize_Scale, self).__init__()
        self.init_norm = init_norm
        self.weight = nn.Parameter(torch.ones(1, dim) * init_norm)

    def forward(self, bottom):
        # input is (n, dim)
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
        concat = torch.cat([self.lfeat_normalizer(lfeats), 
                            self.dif_lfeat_normalizer(dif_lfeats)], 1)
        output = self.fc(concat)
        return output

"""
Takes relative location (n, c, 5) and object features (n, c, 2048) as inputs, 
then output encoded contexts (n, c, 512) and masks (n, c)
"""
class RelationEncoder(nn.Module):
    def __init__(self, opt):
        super(RelationEncoder, self).__init__()
        self.vis_feat_normalizer = Normalize_Scale(opt['fc7_dim'], 
                                                   opt['visual_init_norm'])
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
        # (n * num_cxt, fc7_dim)
        cxt_feats  = self.vis_feat_normalizer(cxt_feats.view(batch*num_cxt, -1)) 
        # (n * num_cxt, 5)
        cxt_lfeats = self.lfeat_normalizer(cxt_lfeats.view(batch*num_cxt, -1))

        # joint embed
        concat = torch.cat([cxt_feats, cxt_lfeats], 1) # (n*num_cxt, fc7_dim+5)
        rel_feats = self.fc(concat)                    # (n * num_cxt, jemb_dim)
        rel_feats = rel_feats.view(batch, num_cxt, -1) # (n, num_cxt, jemb_dim)

        # return
        return rel_feats, masks