import torch
import os
from torch import nn

class AdversarialLoss(nn.Module):
    def __init__(self, args):
        self.args = args

        self.nps_loss = None
        self.tv_loss = None
        self.disp_loss = None
        self.print_colors = self.printibility_colors()
        self.target = None
        self.depth = None
        self.patch = None
        
    def forward(self, patch, depth, target):
        self.target = target
        self.depth = depth
        self.patch = patch
        self.disp_loss = self.calc_disp_loss()
        self.nps_loss = self.calc_nps_loss()
        self.tv_loss = self.calc_tv_loss()
        loss = self.nps_loss + self.tv_loss + self.disp_loss
        return loss.mean()
    

    def printibility_colors(self):
        printable_colors = []
        printable_color_lines = open(os.path.expanduser('~') + '/neural_nemesis/DLG2_p3/src/Src/list/printable30values.txt').readlines()
        for line in printable_color_lines:
            string_color_array = line.strip()
            string_color_array = string_color_array.split(',')

            color_array = []
            for color_float in string_color_array:
                color_array.append(float(color_float))

            printable_colors.append(color_array)
        printable_colors = torch.Tensor(printable_colors).requires_grad_().cuda()
        return printable_colors
        
    ### Both should return tensors of the same size of the img with loss 
    # Non-printability score loss TODO: Implement
    def calc_nps_loss(self):
        return self.patch
    
    # Total Variation Loss TODO: Implement
    def calc_tv_loss(self):
        return self.patch
    
    def calc_disp_loss(self, ):
        return nn.L1Loss(reduction='sum')(self.depth, (self.target))