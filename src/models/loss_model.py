import torch
from torch import nn

class AdversarialLoss(nn.Module):
    def __init__(self, args):
        self.args = args
        self.nps_loss = None
        self.tv_loss = None
        self.disp_loss = None
        
    def forward(self, patch):
        self.disp_loss = self.calc_disp_loss(patch)
        self.nps_loss = self.calc_nps_loss(patch)
        self.tv_loss = self.calc_tv_loss(patch)
        loss = self.nps_loss + self.tv_loss + self.disp_loss
        return loss.mean()
    

    ### Both should return tensors of the same size of the img with loss 
    # Non-printability score loss TODO: Implement
    def calc_nps_loss(self, patch):
        return patch
    
    # Total Variation Loss TODO: Implement
    def calc_tv_loss(self, patch):
        return patch
    
    def calc_disp_loss(self, patch):
        nn.L1Loss(torch.full(patch.size(), 15.0, requires_grad=True), patch, reduction = "sum")
        return patch