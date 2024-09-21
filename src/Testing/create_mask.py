import torch
import argparse
import numpy as np
import time
import os
import torchvision
import PIL
import torchvision.transforms.functional as transform
import torch 
import torchvision 
from torchvision.io import read_image 
import torchvision.transforms as T 
to_image = T.ToPILImage()
img = torchvision.io.read_image("/home/cmfrench/RBE474X/DeepLearningPart2/DLG2_p3/src/baseline_patch.png")


for i in range(img.size(0)):
    for k in range(img.size(1)):
        for l in range(img.size(2)):
            if((k-(img.size(1)/2))**2 + (l-(img.size(2)/2))**2 > (img.size(1)/2)**2):
                img[i, k, l] = 0
            else:
                img[i, k, l] = 255

image = to_image(img)
image.convert("RGB")
image.save("/home/cmfrench/RBE474X/DeepLearningPart2/DLG2_p3/src/mask.png")

