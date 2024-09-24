import torchvision
import PIL
import torchvision.transforms.functional as transform
import random 

import torch 
import torchvision 
from torchvision.io import read_image 
import torchvision.transforms as T 

from PIL import Image
to_image = T.ToPILImage()
img1 = Image.open("/home/cmfrench/RBE474X/DeepLearningPart2/DLG2_p3/src/Dataset/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/0000000000.png")
img2 = Image.open("/home/cmfrench/RBE474X/DeepLearningPart2/DLG2_p3/src/baseline_patch.png")

img1.paste(img2,(int(img1.width/2),int(img1.height/2)))

img1.save("/home/cmfrench/RBE474X/DeepLearningPart2/DLG2_p3/src/Testing/imageinimage.png")

