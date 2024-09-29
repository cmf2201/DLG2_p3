import torch
import numpy as np
from torchvision.transforms import ToPILImage

to_image = ToPILImage()

for i in range(20):
    patch = np.load('/home/ctnguyen/neural_nemesis/DLG2_p3/src/Dst/checkpoints/result/epoch_' + str(i) + '_mask.npy')

    patch = torch.Tensor(patch)

    image = to_image(patch/255)
    image.save('/home/ctnguyen/neural_nemesis/DLG2_p3/src/Testing/generated/patch_' + str(i) + '.png')