from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn

class MaxMarginCriterion(nn.Module):

    def __init__(self, visual_rank_weight, lang_rank_weight, margin):
        super(MaxMarginCriterion, self).__init__()
        self.visual_rank = visual_rank_weight > 0 
        self.lang_rank = lang_rank_weight > 0
        self.visual_rank_weight = visual_rank_weight
        self.lang_rank_weight = lang_rank_weight
        self.margin = margin

    def forward(self, cossim):
        N = cossim.size(0)
        batch_size = 0
        if self.visual_rank and not self.lang_rank:
            batch_size = N//2
            assert isinstance(batch_size, int)
            paired = cossim[:batch_size]
            unpaired = cossim[batch_size:]
            visual_rank_loss = self.visual_rank_weight * torch.clamp(self.margin + unpaired - paired, min=0)
            lang_rank_loss = 0.
            
        elif not self.visual_rank and self.lang_rank:
            batch_size = N//2
            assert isinstance(batch_size, int)
            paird = cossim[:batch_size]
            unpaired = cossim[batch_size:]
            lang_rank_loss = self.lang_rank_weight * torch.clamp(self.margin + unpaired - paired, min=0)
            visual_rank_loss = 0.

        elif self.visual_rank and self.lang_rank:
            batch_size = N//3
            assert isinstance(batch_size, int)
            paired = cossim[:batch_size]
            visual_unpaired = cossim[batch_size: batch_size*2]
            lang_unpaired = cossim[batch_size*2:]
            visual_rank_loss = self.visual_rank_weight * torch.clamp(self.margin + visual_unpaired - paired, 0)
            lang_rank_loss = self.lang_rank_weight * torch.clamp(self.margin + lang_unpaired - paired, 0)

        else:
            raise NotImplementedError

        loss = (visual_rank_loss + lang_rank_loss).sum() / batch_size
        return loss

