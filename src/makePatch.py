import torch
import torchvision
import argparse
import numpy as np
import time
import os
# from models.adversarial_models import AdversarialModels
# from models.loss_model import AdversarialLoss
from torchvision.transforms import v2
from torchvision.transforms import ToPILImage
# from torchvision.transforms.functional import pil_to_tensor
# from torchvision.transforms.functional import crop
# from utils.dataloader import LoadFromImageFile
# from utils.utils import *
# from tqdm import tqdm
# import warnings
from PIL import Image
# import matplotlib.pyplot as plt

# warnings.simplefilter('ignore')

parser = argparse.ArgumentParser(description='Generating Adversarial Patches')
parser.add_argument('--data_root', type=str, help='path to dataset', default='dataset')
parser.add_argument('--train_list', type=str, default='Src/list/eigen_train_list.txt')
parser.add_argument('--print_file', type=str, default='Src/list/printable30values.txt')
parser.add_argument('--distill_ckpt', type=str, default="repository/release-StereoUnsupFt-Mono-pt-CK.ckpt")
parser.add_argument('--height', type=int, help='input image height', default=256)
parser.add_argument('--width', type=int, help='input image width', default=512)
parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size', default=16)
parser.add_argument('-j', '--num_threads', type=int, help='data loading workers', default=0)
parser.add_argument('--lr', type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--num_epochs', type=int, help='number of total epochs', default=20)
parser.add_argument('--seed', type=int, help='seed for random functions, and network initialization', default=0)
parser.add_argument('--patch_size', type=int, help='Resolution of patch', default=56)
parser.add_argument('--patch_shape', type=str, help='circle or square', default='circle')
parser.add_argument('--patch_path', type=str, help='Initialize patch from file', default="/home/cmfrench/RBE474X/DLG2_p3/src/baseline_patch.png")
parser.add_argument('--mask_path', type=str, help='Initialize mask from file', default="/home/cmfrench/RBE474X/DLG2_p3/src/mask.png")
parser.add_argument('--colors_path', type=str, help='Directory of printable colors', default="/home/cmfrench/RBE474X/DLG2_p3/src/Src/list/printable30values.txt")
parser.add_argument('--target_disp', type=int, default=120)
parser.add_argument('--model', nargs='*', type=str, default='distill', choices=['distill'], help='Model architecture')
parser.add_argument('--name', type=str, help='output directory', default="result")
args = parser.parse_args()

to_image = ToPILImage()

# setup your torchvision/anyother transforms here. This is for adding noise/perspective transforms and other changes to the background
# train_transform = None

# train_set = LoadFromImageFile(
#     args.data_root,
#     args.train_list,
#     mask_path=args.mask_path,
#     seed=args.seed,
#     train=True,
#     monocular=True,
#     transform=train_transform,
#     extension=".png"
# )

# train_loader = torch.utils.data.DataLoader(
#     dataset=train_set,
#     batch_size=args.batch_size,
#     shuffle=True,
#     num_workers=args.num_threads,
#     pin_memory=True,
#     drop_last=True
# )

# Transforms
transforms = v2.Compose([
    v2.RandomResize(30,70),
    v2.RandomPerspective(),
    v2.RandomRotation(degrees=(-15,15)),
    v2.RandomPhotometricDistort()
])

print('===============================')

# Patch and Mask
# Initialize a random patch image, resize to fix within training images
patch_cpu = torchvision.io.read_image(args.patch_path)
mask_cpu = torchvision.io.read_image(args.mask_path)
# mask_cpu = v2.Resize(size=(56,56))(mask_cpu).requires_grad_()

for i in range(1):
    img = to_image(patch_cpu)
    img = img.convert("RGBA")

    # img = Image.composite(img,img,to_image(mask_cpu).convert("L"))
    img2 = to_image(mask_cpu).convert("L")
    img = img.paste(img,(0,0),img2)
    # img2.save(f"distortions/test2.png")
    # # img = transforms(patch_cpu)
    # # img = to_image(img.detach())
    # img.

    img.save(f"distortions/test.png")