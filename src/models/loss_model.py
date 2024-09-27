import torch
from torch import nn

class AdversarialLoss(nn.Module):
    def __init__(self, args):
        self.args = args
        self.nps_loss = None
        self.tv_loss = None
        
    def forward(self, patch, patch_loss):
        self.nps_loss = self.calc_nps_loss(patch)
        self.tv_loss = self.calc_tv_loss(patch)
        disp_loss = patch_loss
        loss = self.nps_loss + self.tv_loss + disp_loss
        return loss.mean()
    

    ### Both should return tensors of the same size of the img with loss 
    # Non-printability score loss TODO: Implement
    def calc_nps_loss(self, patch):
        return patch
    
    # Total Variation Loss TODO: Implement
    def calc_tv_loss(self, patch):
        return patch