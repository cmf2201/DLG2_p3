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


