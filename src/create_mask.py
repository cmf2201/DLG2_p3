import torch
import argparse
import cv2
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

img = torchvision.io.read_image('baseline_patch.png')
print(img)

for i in range(img.size(0)):
    for k in range(img.size(1)):
        for l in range(img.size(2)):
            if(img[k-112]**2 + img[l-112]**2 > 112**2):
                img[i, k, l] = 0

print(img)

