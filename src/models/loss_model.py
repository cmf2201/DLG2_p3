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
        
    def forward(self, depth, target): #include patch
        self.target = target
        self.depth = depth
        self.patch = depth
        self.disp_loss = self.calc_disp_loss()
        # self.nps_loss = self.calc_nps_loss()
        # self.tv_loss = self.calc_tv_loss()
        loss = self.disp_loss ## TODO:FIX WITH OTHERS LOSSES
        return loss.mean()
    

    def printibility_colors(self):
        printable_colors = []
        printable_color_lines = open(self.args.colors_path).readlines()
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
        # assuming self.patch is patch.png 3x56x56
        new_patch = self.patch.permute(1,2,0) # now (H, W, C)
        squeezed_patch = new_patch.reshape(-1,3) # shape = (H*W, 3)
        distances = torch.cdist(squeezed_patch.float(), self.print_colors) # calculates Euclidean distance between each color value for each pixel
        min_distances = torch.min(distances, dim = 1).values() # finds min distance value for each pixel
        min_distances_final = min_distances.view(new_patch.shape[0], new_patch.shape[1], 1).repeat(1, 1, 3) # reshapes back to (H, W, C)
        loss = min_distances_final.permute(0, 1, 2) # back to (C, H, W)
        target_tensor = torch.zeros_like(self.patch) # assuming target minimum distance is 0
        return nn.L1loss(reduction='sum')(loss, target_tensor)
         
    
    # Total Variation Loss TODO: Implement
    def calc_tv_loss(self):
        return self.patch
    
    def calc_disp_loss(self):
        return nn.L1Loss(reduction='sum')(self.depth, (self.target))
